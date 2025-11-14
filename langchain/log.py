import json
import logging
from pathlib import Path


def classify_event(message: str) -> dict[str, str] | None:
    """Return a structured representation for the log lines we care about."""
    text = (message or "").strip()

    static_events = {
        "=== AGENT START ===": "agent_start",
        "=== AGENT END ===": "agent_end",
        "--- BEFORE MODEL ---": "before_model",
        "--- AFTER MODEL ---": "after_model",
    }
    if text in static_events:
        return {"event": static_events[text]}

    if text.startswith("Initial state:"):
        return {"event": "initial_state", "content": text.split(":", 1)[1].strip()}

    if text.startswith("Final messages:"):
        return {"event": "final_messages"}

    if text.startswith("Model output (ai):"):
        content = text.split(":", 1)[1].strip()
        return {"event": "model_output", "role": "ai", "content": content}

    if text.startswith("human:"):
        return {"event": "message", "role": "human", "content": text.split(":", 1)[1].strip()}

    if text.startswith("ai:"):
        return {"event": "message", "role": "ai", "content": text.split(":", 1)[1].strip()}

    if text.startswith("tool:"):
        return {"event": "tool_message", "content": text.split(":", 1)[1].strip()}

    if text.startswith(">>> TOOL CALL"):
        _, payload = text.split(": ", 1)
        name, args = payload.split("(", 1)
        return {
            "event": "tool_call",
            "tool": name.strip(),
            "args": args.rstrip(")"),
        }

    if text.startswith("<<< TOOL RESULT"):
        prefix = "<<< TOOL RESULT (async) from "
        after_prefix = text[len(prefix):]
        tool_name, data = after_prefix.split(": ", 1)
        return {
            "event": "tool_result",
            "tool": tool_name.strip(),
            "result": data.strip(),
        }

    return None


class StructuredEventFilter(logging.Filter):
    """Allow only classified events to reach the JSON file handler."""
    def filter(self, record: logging.LogRecord) -> bool:
        payload = classify_event(record.getMessage())
        if not payload:
            return False
        record.structured_payload = payload
        return True


class JSONFormatter(logging.Formatter):
    """Serialize filtered records as JSON with only the allowed fields."""
    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "structured_payload", None) or {"event": "log", "message": record.getMessage()}
        return json.dumps(payload, ensure_ascii=False)


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "agent_run.json"

file_handler = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
file_handler.addFilter(StructuredEventFilter())
file_handler.setFormatter(JSONFormatter())

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

logger = logging.getLogger("agent_logger")

for name in ["httpx", "httpcore", "urllib3", "openai", "langchain", "langgraph", "mcp"]:
    logging.getLogger(name).setLevel(logging.WARNING)