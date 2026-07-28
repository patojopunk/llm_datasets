from __future__ import annotations

"""One request-level observer for LLM and tool activity.

This file is intentionally the main observability hook for agent execution.
LangChain calls these callback methods when the model and tools actually run.
That means we do not estimate LLM calls from message history and we do not
wrap/copy every tool object.
"""

import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import trace

from backend.observability.models import ToolCallRecord, ToolCallStarted, ToolSummary
from backend.observability.otel import get_tracer

TRACER = get_tracer(__name__)


class RequestObserver(BaseCallbackHandler):
    """Collect real LLM and tool events for one agent request."""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.tool_records: list[ToolCallRecord] = []

        self._seen_llm_runs: set[str] = set()
        self._active_tools: dict[str, ToolCallStarted] = {}
        self._active_tool_spans: dict[str, trace.Span] = {}
        self._next_tool_index = 0

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        """Called by LangChain when a chat model call starts."""
        self._record_llm_start(run_id=run_id, model_type="chat_model")

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        """Called by LangChain when a non-chat LLM call starts."""
        self._record_llm_start(run_id=run_id, model_type="llm")

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        """Called by LangChain when a tool starts running."""
        run_key = str(run_id)
        tool_name = str(serialized.get("name") or "unknown_tool")
        input_size = len(input_str or "")
        tool_index = self._next_tool_index
        self._next_tool_index += 1

        self._active_tools[run_key] = ToolCallStarted(
            index=tool_index,
            name=tool_name,
            run_id=run_key,
            input_size=input_size,
            start_time_s=time.perf_counter(),
        )

        span = TRACER.start_span(f"tool.invoke {tool_name}")
        span.set_attribute("tool.index", tool_index)
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.run_id", run_key)
        span.set_attribute("tool.input.size", input_size)
        self._active_tool_spans[run_key] = span

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        """Called by LangChain when a tool finishes successfully."""
        run_key = str(run_id)
        started = self._active_tools.pop(run_key, None)
        span = self._active_tool_spans.pop(run_key, None)

        if started is None:
            return

        duration_ms = self._duration_ms(started.start_time_s)
        output_size = len(str(output))

        self.tool_records.append(
            ToolCallRecord(
                index=started.index,
                name=started.name,
                run_id=started.run_id,
                success=True,
                duration_ms=duration_ms,
                input_size=started.input_size,
                output_size=output_size,
            )
        )

        if span is not None:
            span.set_attribute("tool.success", True)
            span.set_attribute("tool.duration_ms", duration_ms)
            span.set_attribute("tool.output.size", output_size)
            span.end()

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        """Called by LangChain when a tool raises an error."""
        run_key = str(run_id)
        started = self._active_tools.pop(run_key, None)
        span = self._active_tool_spans.pop(run_key, None)

        if started is None:
            return

        duration_ms = self._duration_ms(started.start_time_s)
        error_message = str(error)[:500]

        self.tool_records.append(
            ToolCallRecord(
                index=started.index,
                name=started.name,
                run_id=started.run_id,
                success=False,
                duration_ms=duration_ms,
                input_size=started.input_size,
                output_size=0,
                error_type=type(error).__name__,
                error_message=error_message,
            )
        )

        if span is not None:
            span.set_attribute("tool.success", False)
            span.set_attribute("tool.duration_ms", duration_ms)
            span.set_attribute("error.type", type(error).__name__)
            span.set_attribute("error.message", error_message)
            span.record_exception(error)
            span.end()

    def tools_used(self) -> list[str]:
        return [record.name for record in self.tool_records]

    def tool_summary(self) -> ToolSummary:
        names = self.tools_used()
        return ToolSummary(
            call_count=len(self.tool_records),
            total_duration_ms=sum(record.duration_ms for record in self.tool_records),
            max_duration_ms=max((record.duration_ms for record in self.tool_records), default=0.0),
            names=names,
            unique_names=list(dict.fromkeys(names)),
            success_count=sum(1 for record in self.tool_records if record.success),
            error_count=sum(1 for record in self.tool_records if not record.success),
        )

    def add_tool_events_to_span(self, span: trace.Span) -> None:
        """Attach a compact timeline of completed tool calls to a parent span."""
        for record in self.tool_records:
            span.add_event(
                "tool.completed",
                {
                    "tool.index": record.index,
                    "tool.name": record.name,
                    "tool.run_id": record.run_id,
                    "tool.success": record.success,
                    "tool.duration_ms": record.duration_ms,
                },
            )

    def _record_llm_start(self, *, run_id: Any, model_type: str) -> None:
        run_key = str(run_id)

        if run_key in self._seen_llm_runs:
            return

        self._seen_llm_runs.add(run_key)
        self.llm_calls += 1

        span = trace.get_current_span()
        if span.is_recording():
            span.add_event(
                "llm.started",
                {
                    "llm.call_count": self.llm_calls,
                    "llm.model_type": model_type,
                    "llm.run_id": run_key,
                },
            )

    @staticmethod
    def _duration_ms(start_time_s: float) -> float:
        return (time.perf_counter() - start_time_s) * 1000.0
