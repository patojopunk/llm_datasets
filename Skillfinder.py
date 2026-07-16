# backend/observability/request_observer.py

from pathlib import Path
import json


def _extract_skill_from_read_file(self, *, tool_name: str, input_str: str) -> str | None:
    if tool_name != "read_file":
        return None

    # Usually input_str is JSON-ish, like {"file_path": "skills/math/SKILL.md"}
    try:
        data = json.loads(input_str)
        path = data.get("file_path") or data.get("path")
    except Exception:
        path = input_str

    if not path:
        return None

    path_text = str(path).replace("\\", "/")

    if not path_text.endswith("SKILL.md"):
        return None

    # skills/math/SKILL.md -> math
    return Path(path_text).parent.name



# on_tool_start

skill_name = self._extract_skill_from_read_file(
    tool_name=tool_name,
    input_str=input_str,
)

if skill_name:
    self._emit_stream_event(
        "skill_selected",
        {
            "message": f"Selected skill: {skill_name}",
            "skill_name": skill_name,
        },
    )
