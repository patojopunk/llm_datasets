from __future__ import annotations

"""FastAPI entry point for the slim observability app."""

import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastmcp.utilities.lifespan import combine_lifespans
from opentelemetry import trace

from backend.agent import AgentEngine
from backend.config import Settings
from backend.mcp_runtime import MCPRuntime
from backend.observability.metrics import JSONLMetricsWriter
from backend.observability.models import ChatTelemetryEvent, EngineMetrics
from backend.observability.otel import current_trace_id, instrument_fastapi, setup_otel
from backend.schemas import ChatRequest, ChatResponse, ChatResponseMetrics
from backend.session_store import ChatSessionStore
from gateways.core_gateway import build_gateway


async def get_sessions(app: FastAPI) -> ChatSessionStore:
    """Create the MCP runtime, agent engine, and sessions on first use.

    This is intentionally lazy because the MCP gateway is mounted inside the
    same FastAPI app at /mcp. The app must finish startup before the runtime
    tries to load tools from http://127.0.0.1:8000/mcp.
    """
    if app.state.sessions is not None:
        return app.state.sessions

    async with app.state.agent_start_lock:
        if app.state.sessions is not None:
            return app.state.sessions

        runtime = MCPRuntime(settings=app.state.settings)
        await runtime.start()

        engine = AgentEngine(settings=app.state.settings, mcp_runtime=runtime)
        await engine.start()

        app.state.mcp_runtime = runtime
        app.state.engine = engine
        app.state.sessions = ChatSessionStore(engine=engine)

        return app.state.sessions


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI app and wire runtime dependencies."""
    settings = settings or Settings.from_env()
    setup_otel(
        enabled=settings.otel_enabled,
        service_name=settings.otel_service_name,
        otlp_endpoint=settings.otlp_endpoint,
    )

    gateway = build_gateway()
    gateway_app = gateway.http_app(path="/")

    @asynccontextmanager
    async def app_lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.mcp_runtime = None
        app.state.engine = None
        app.state.sessions = None
        app.state.agent_start_lock = asyncio.Lock()
        app.state.metrics = JSONLMetricsWriter(settings.metrics_path)

        try:
            yield
        finally:
            if app.state.mcp_runtime is not None:
                await app.state.mcp_runtime.stop()

    app = FastAPI(
        title="FastMCP Observability Callback Simple",
        lifespan=combine_lifespans(app_lifespan, gateway_app.lifespan),
    )

    instrument_fastapi(app, enabled=settings.otel_enabled)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount("/mcp", gateway_app)

    @app.get("/health")
    async def health() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "agent_started": app.state.sessions is not None,
        }

    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        """Run a chat turn and record request-level telemetry."""
        req_t0 = time.perf_counter()
        reply = ""
        tools_used: list[str] = []
        engine_metrics = EngineMetrics()
        success = False
        error: str | None = None
        trace_id: str | None = None

        request_span = trace.get_current_span()
        request_span.set_attribute("chat.session_id", req.session_id)
        request_span.set_attribute("chat.user_message_len", len(req.message))

        try:
            sessions = await get_sessions(app)
            result = await sessions.chat(req.session_id, req.message)
            reply = result.reply
            tools_used = result.tools_used
            engine_metrics = result.metrics
            trace_id = result.trace_id
            success = True
        except Exception as exc:
            error = str(exc)
            trace_id = current_trace_id()
            request_span.set_attribute("chat.error_type", type(exc).__name__)
            raise
        finally:
            total_ms = (time.perf_counter() - req_t0) * 1000.0
            request_span.set_attribute("chat.success", success)
            request_span.set_attribute("chat.total_response_time_ms", total_ms)
            request_span.set_attribute("chat.tools_used", ",".join(tools_used))
            request_span.set_attribute("chat.tool_call_count", engine_metrics.tool_calls)
            request_span.set_attribute("chat.tool_total_duration_ms", engine_metrics.tool_total_duration_ms)

            app.state.metrics.write(
                ChatTelemetryEvent(
                    session_id=req.session_id,
                    trace_id=trace_id,
                    user_message_len=len(req.message),
                    tools_used=tools_used,
                    success=success,
                    error=error,
                    total_response_time_ms=total_ms,
                    engine_metrics=engine_metrics,
                )
            )

        return ChatResponse(
            reply=reply,
            tools_used=tools_used,
            mode="deep_agent",
            trace_id=trace_id,
            metrics=ChatResponseMetrics(
                total_response_time_ms=total_ms,
                engine_time_ms=engine_metrics.engine_time_ms,
                llm_calls=engine_metrics.llm_calls,
                tool_calls=engine_metrics.tool_calls,
                tool_total_duration_ms=engine_metrics.tool_total_duration_ms,
                tool_max_duration_ms=engine_metrics.tool_max_duration_ms,
                tool_success_count=engine_metrics.tool_success_count,
                tool_error_count=engine_metrics.tool_error_count,
                lock_wait_ms=engine_metrics.lock_wait_ms,
            ),
        )

    return app


app = create_app()
