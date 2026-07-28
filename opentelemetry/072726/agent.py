from __future__ import annotations

"""Agent engine for the backend.

The engine owns the shared Deep Agent and uses RequestObserver to collect real
LLM/tool events during one chat turn. The normal /chat path still uses ainvoke().
The /chat/stream path uses native agent.astream() so the API can show live
agent flow, tool calls, tool results, and final-answer tokens.
"""

import json
import re
import time
from pathlib import PurePath
from typing import Any, AsyncIterator

from deepagents import create_deep_agent
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
    convert_to_messages,
)
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from backend.config import Settings
from backend.mcp_runtime import MCPRuntime
from backend.observability.models import EngineMetrics
from backend.observability.otel import get_tracer
from backend.observability.request_observer import RequestObserver

TRACER = get_tracer(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools and skills. "
    "Use tools when they improve correctness, and do not invent tool results."
)

SKILL_MD_PATTERN = re.compile(r"(?P<path>[^\"'{}\s]+[/\\]SKILL\.md)", re.IGNORECASE)


class EngineResult(BaseModel):
    """Structured result returned by the agent engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reply: str = ""
    tools_used: list[str] = Field(default_factory=list)
    skills_used: list[str] = Field(default_factory=list)
    metrics: EngineMetrics = Field(default_factory=EngineMetrics)
    state: dict[str, Any] = Field(default_factory=dict)


class AgentEngine:
    """Create and run the shared Deep Agent."""

    def __init__(self, settings: Settings, mcp_runtime: MCPRuntime) -> None:
        self.settings = settings
        self.mcp_runtime = mcp_runtime
        self.agent: Any | None = None

    async def start(self) -> None:
        """Load MCP tools and build the agent once during app startup."""
        with TRACER.start_as_current_span("engine.start") as span:
            tools = await self.mcp_runtime.get_all_tools()
            span.set_attribute("tool.count", len(tools))
            span.set_attribute("tool.names", ",".join(tool.name for tool in tools))
            span.set_attribute("skills.dir", self.settings.skills_dir)

            self.agent = create_deep_agent(
                model=ChatOpenAI(
                    model=self.settings.llm_model,
                    base_url=self.settings.llm_base_url,
                    api_key=self.settings.llm_api_key,
                    temperature=self.settings.llm_temperature,
                ),
                tools=tools,
                system_prompt=self.settings.app_system_prompt or DEFAULT_SYSTEM_PROMPT,
                # Skills are built into this app. Deep Agents handles discovery
                # and progressive loading from the SKILL.md files.
                skills=[self.settings.skills_dir],
            )

    async def respond(self, *, session_id: str, state: dict[str, Any], user_text: str) -> EngineResult:
        """Run one chat turn and return the reply, updated state, and metrics."""
        if self.agent is None:
            raise RuntimeError("AgentEngine.start() must be called before respond().")

        agent_input = self._build_agent_input(state=state, user_text=user_text)
        prior_messages = state.get("messages", [])

        with TRACER.start_as_current_span("engine.respond") as span:
            span.set_attribute("session.id", session_id)
            span.set_attribute("message.history_count", len(prior_messages))
            span.set_attribute("message.user_text_len", len(user_text))

            observer = RequestObserver()
            start_time = time.perf_counter()

            with TRACER.start_as_current_span("engine.agent_invoke") as invoke_span:
                invoke_span.set_attribute("session.id", session_id)
                raw_result = await self.agent.ainvoke(
                    agent_input,
                    config={
                        "configurable": {"thread_id": session_id},
                        "callbacks": [observer],
                    },
                )

            engine_time_ms = (time.perf_counter() - start_time) * 1000.0
            return self._build_engine_result(
                raw_result=raw_result,
                observer=observer,
                engine_time_ms=engine_time_ms,
                span=span,
            )

    async def stream_respond(
        self,
        *,
        session_id: str,
        state: dict[str, Any],
        user_text: str,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one chat turn using native agent.astream().

        This method yields small plain dict events that the FastAPI layer turns
        into Server-Sent Events. It also keeps RequestObserver attached so the
        final metrics still come from real LangChain callbacks.
        """
        if self.agent is None:
            raise RuntimeError("AgentEngine.start() must be called before stream_respond().")

        agent_input = self._build_agent_input(state=state, user_text=user_text)
        prior_messages = state.get("messages", [])

        with TRACER.start_as_current_span("engine.stream_respond") as span:
            span.set_attribute("session.id", session_id)
            span.set_attribute("message.history_count", len(prior_messages))
            span.set_attribute("message.user_text_len", len(user_text))

            observer = RequestObserver()
            start_time = time.perf_counter()
            latest_state: dict[str, Any] | None = None
            stream_state: dict[str, Any] = {
                "tool_names_by_key": {},
                "tool_args_by_key": {},
                "announced_tool_keys": set(),
                "announced_skill_names": set(),
                "announced_skill_order": [],
            }

            yield {
                "type": "flow",
                "message": "Starting native agent.astream run.",
                "source": "main",
            }

            with TRACER.start_as_current_span("engine.agent_astream") as invoke_span:
                invoke_span.set_attribute("session.id", session_id)
                invoke_span.set_attribute("stream.mode", "updates,messages,values")
                invoke_span.set_attribute("stream.version", "v2")

                async for chunk in self.agent.astream(
                    agent_input,
                    config={
                        "configurable": {"thread_id": session_id},
                        "callbacks": [observer],
                    },
                    stream_mode=["updates", "messages", "values"],
                    subgraphs=True,
                    version="v2",
                ):
                    if self._is_values_chunk(chunk):
                        latest_state = self._chunk_data(chunk)
                        continue

                    for event in self._events_from_stream_chunk(chunk, stream_state):
                        yield event

            engine_time_ms = (time.perf_counter() - start_time) * 1000.0
            raw_result = latest_state or {"messages": agent_input["messages"]}
            result = self._build_engine_result(
                raw_result=raw_result,
                observer=observer,
                engine_time_ms=engine_time_ms,
                span=span,
            )
            result.skills_used.extend(self._skills_from_stream_state(stream_state))

            yield {
                "type": "final",
                "message": "Agent run finished.",
                "reply": result.reply,
                "tools_used": result.tools_used,
                "skills_used": result.skills_used,
                "metrics": result.metrics.model_dump(),
                "elapsed_ms": engine_time_ms,
                "_result": result,
            }

    def _build_engine_result(
        self,
        *,
        raw_result: dict[str, Any],
        observer: RequestObserver,
        engine_time_ms: float,
        span: Any,
    ) -> EngineResult:
        """Build EngineResult and attach the same observability attributes."""
        messages = convert_to_messages(raw_result.get("messages", []))
        reply = self._last_ai_message_text(messages)
        tool_summary = observer.tool_summary()

        if span is not None:
            span.set_attribute("engine.time_ms", engine_time_ms)
            span.set_attribute("llm.calls", observer.llm_calls)
            span.set_attribute("reply.length", len(reply))

            for name, value in tool_summary.to_span_attributes().items():
                span.set_attribute(name, value)

            observer.add_tool_events_to_span(span)

        return EngineResult(
            reply=reply,
            tools_used=observer.tools_used(),
            metrics=EngineMetrics(
                engine_time_ms=engine_time_ms,
                llm_calls=observer.llm_calls,
                tool_calls=tool_summary.call_count,
                tool_total_duration_ms=tool_summary.total_duration_ms,
                tool_max_duration_ms=tool_summary.max_duration_ms,
                tool_success_count=tool_summary.success_count,
                tool_error_count=tool_summary.error_count,
            ),
            state={"messages": messages},
        )

    @staticmethod
    def _build_agent_input(*, state: dict[str, Any], user_text: str) -> dict[str, Any]:
        prior_messages = state.get("messages", [])
        return {
            "messages": [
                *prior_messages,
                {"role": "user", "content": user_text},
            ]
        }

    def _events_from_stream_chunk(
        self,
        chunk: Any,
        stream_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Convert one LangGraph v2 StreamPart into small UI-safe events."""
        if not isinstance(chunk, dict):
            return [
                {
                    "type": "stream_debug",
                    "message": "Received a legacy stream chunk shape.",
                    "data_preview": self._preview(chunk),
                }
            ]

        chunk_type = str(chunk.get("type", ""))
        if chunk_type == "updates":
            return self._events_from_update_chunk(chunk)
        if chunk_type == "messages":
            return self._events_from_message_chunk(chunk, stream_state)
        if chunk_type == "custom":
            return [
                {
                    "type": "custom",
                    "message": "Custom stream update.",
                    "source": self._source_from_namespace(chunk.get("ns")),
                    "data": self._json_safe(chunk.get("data")),
                }
            ]
        return []

    def _events_from_update_chunk(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        data = chunk.get("data")
        if not isinstance(data, dict):
            return []

        source = self._source_from_namespace(chunk.get("ns"))
        events: list[dict[str, Any]] = []
        for node_name in data:
            if node_name.startswith("__"):
                continue
            events.append(
                {
                    "type": "flow",
                    "message": f"{source} step: {node_name}",
                    "source": source,
                    "node": node_name,
                    "namespace": self._namespace_list(chunk.get("ns")),
                }
            )
        return events

    def _events_from_message_chunk(
        self,
        chunk: dict[str, Any],
        stream_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        data = chunk.get("data")
        if not isinstance(data, tuple) or len(data) != 2:
            return []

        token, metadata = data
        source = self._source_from_namespace(chunk.get("ns"))
        namespace = self._namespace_list(chunk.get("ns"))

        if isinstance(token, AIMessageChunk):
            return self._events_from_ai_message_chunk(
                token=token,
                metadata=metadata,
                source=source,
                namespace=namespace,
                stream_state=stream_state,
            )

        if isinstance(token, ToolMessage):
            tool_name = token.name or "unknown_tool"
            content_preview = "[file content omitted]" if tool_name == "read_file" else self._preview(token.content)
            return [
                {
                    "type": "tool_result",
                    "message": f"Tool result received: {tool_name}",
                    "tool_name": tool_name,
                    "source": source,
                    "namespace": namespace,
                    "content_preview": content_preview,
                }
            ]

        return []

    def _events_from_ai_message_chunk(
        self,
        *,
        token: AIMessageChunk,
        metadata: Any,
        source: str,
        namespace: list[str],
        stream_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []

        if token.tool_call_chunks:
            for tool_call in token.tool_call_chunks:
                events.extend(
                    self._events_from_tool_call_chunk(
                        tool_call=tool_call,
                        source=source,
                        namespace=namespace,
                        stream_state=stream_state,
                    )
                )
            return events

        text = self._message_content_text(token.content)
        if text:
            events.append(
                {
                    "type": "token",
                    "text": text,
                    "source": source,
                    "namespace": namespace,
                    "metadata": self._small_metadata(metadata),
                }
            )

        return events

    def _events_from_tool_call_chunk(
        self,
        *,
        tool_call: dict[str, Any],
        source: str,
        namespace: list[str],
        stream_state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        call_key = self._tool_call_key(tool_call=tool_call, source=source)
        tool_name = tool_call.get("name") or stream_state["tool_names_by_key"].get(call_key)
        args_delta = tool_call.get("args") or ""
        events: list[dict[str, Any]] = []

        if tool_name:
            stream_state["tool_names_by_key"][call_key] = tool_name

        if tool_name and call_key not in stream_state["announced_tool_keys"]:
            stream_state["announced_tool_keys"].add(call_key)
            events.append(
                {
                    "type": "tool_start",
                    "message": f"Calling tool: {tool_name}",
                    "tool_name": tool_name,
                    "tool_call_id": call_key,
                    "source": source,
                    "namespace": namespace,
                }
            )

        if args_delta:
            existing_args = stream_state["tool_args_by_key"].get(call_key, "")
            args_so_far = existing_args + str(args_delta)
            stream_state["tool_args_by_key"][call_key] = args_so_far

            events.append(
                {
                    "type": "tool_args",
                    "message": f"Tool arguments streaming: {tool_name or 'unknown_tool'}",
                    "tool_name": tool_name or "unknown_tool",
                    "tool_call_id": call_key,
                    "args_delta": str(args_delta),
                    "source": source,
                    "namespace": namespace,
                }
            )

            skill_name = self._skill_from_read_file_args(tool_name=tool_name, args_text=args_so_far)
            if skill_name and skill_name not in stream_state["announced_skill_names"]:
                stream_state["announced_skill_names"].add(skill_name)
                stream_state["announced_skill_order"].append(skill_name)
                events.append(
                    {
                        "type": "skill_selected",
                        "message": f"Selected skill: {skill_name}",
                        "skill_name": skill_name,
                        "reason": "read_file_SKILL_md",
                        "source": source,
                        "namespace": namespace,
                    }
                )

        return events

    @staticmethod
    def _tool_call_key(*, tool_call: dict[str, Any], source: str) -> str:
        call_id = tool_call.get("id")
        if call_id:
            return str(call_id)
        index = tool_call.get("index", "unknown")
        return f"{source}:{index}"

    @staticmethod
    def _skill_from_read_file_args(*, tool_name: str | None, args_text: str) -> str | None:
        if tool_name != "read_file" or "SKILL.md" not in args_text:
            return None

        path_text: str | None = None
        try:
            parsed = json.loads(args_text)
            if isinstance(parsed, dict):
                raw_path = parsed.get("file_path") or parsed.get("path")
                if raw_path:
                    path_text = str(raw_path)
        except json.JSONDecodeError:
            match = SKILL_MD_PATTERN.search(args_text)
            if match:
                path_text = match.group("path")

        if not path_text:
            return None

        normalized = path_text.replace("\\", "/")
        if not normalized.endswith("SKILL.md"):
            return None

        return PurePath(normalized).parent.name

    @staticmethod
    def _skills_from_stream_state(stream_state: dict[str, Any]) -> list[str]:
        return list(stream_state.get("announced_skill_order", []))

    @staticmethod
    def _is_values_chunk(chunk: Any) -> bool:
        return isinstance(chunk, dict) and chunk.get("type") == "values" and isinstance(chunk.get("data"), dict)

    @staticmethod
    def _chunk_data(chunk: dict[str, Any]) -> dict[str, Any]:
        data = chunk.get("data")
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _last_ai_message_text(messages: list[BaseMessage]) -> str:
        """Return the text from the last assistant message."""
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message.text.strip()
        return ""

    @staticmethod
    def _source_from_namespace(namespace: Any) -> str:
        parts = AgentEngine._namespace_list(namespace)
        if not parts:
            return "main"
        return next((part for part in parts if part.startswith("tools:")), "subgraph")

    @staticmethod
    def _namespace_list(namespace: Any) -> list[str]:
        if namespace is None:
            return []
        if isinstance(namespace, list):
            return [str(item) for item in namespace]
        if isinstance(namespace, tuple):
            return [str(item) for item in namespace]
        return [str(namespace)]

    @staticmethod
    def _message_content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)

    @staticmethod
    def _small_metadata(metadata: Any) -> dict[str, Any]:
        if not isinstance(metadata, dict):
            return {}

        keys = ["langgraph_node", "tags", "run_id"]
        return {key: AgentEngine._json_safe(metadata[key]) for key in keys if key in metadata}

    @staticmethod
    def _preview(value: Any, limit: int = 300) -> str:
        text = str(value)
        if len(text) <= limit:
            return text
        return text[:limit] + "..."

    @staticmethod
    def _json_safe(value: Any) -> Any:
        try:
            json.dumps(value)
            return value
        except TypeError:
            pass

        if isinstance(value, BaseMessage):
            return {
                "type": value.__class__.__name__,
                "content": AgentEngine._message_content_text(value.content),
            }

        if hasattr(value, "model_dump"):
            return AgentEngine._json_safe(value.model_dump())

        if isinstance(value, dict):
            return {str(key): AgentEngine._json_safe(item) for key, item in value.items()}

        if isinstance(value, list):
            return [AgentEngine._json_safe(item) for item in value]

        if isinstance(value, tuple):
            return [AgentEngine._json_safe(item) for item in value]

        return str(value)
