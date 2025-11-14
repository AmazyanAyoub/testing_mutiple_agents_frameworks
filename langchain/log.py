import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "agent_run.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        # If you don't want console logs, comment this out:
        logging.StreamHandler()
    ],
)

logger = logging.getLogger("agent_logger")

for name in ["httpx", "httpcore", "urllib3", "openai", "langchain", "langgraph", "mcp"]:
    logging.getLogger(name).setLevel(logging.WARNING)
