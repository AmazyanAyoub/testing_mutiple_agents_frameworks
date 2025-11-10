from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from typing import Any, Callable

from langchain.agents.middleware import ToolRetryMiddleware

from log import logger

retry_failed_tools = ToolRetryMiddleware(
    max_retries=3,
    backoff_factor=2.0,
    initial_delay=1.0,
    max_delay=60.0,
    jitter=True,
)

class FullLoggingMiddleware(AgentMiddleware):
    # Called once per invocation
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        logger.info("=== AGENT START ===")
        logger.info("Initial state: %s", state)
        return None

    # Before every model call
    def before_model(self, state, runtime):
        logger.info("--- BEFORE MODEL ---")
        logger.info("Messages (%d):", len(state["messages"]))
        for m in state["messages"]:
            logger.info("%s: %s", m.type, m.content)
        return None

    # After every model call
    def after_model(self, state, runtime):
        last = state["messages"][-1]
        logger.info("--- AFTER MODEL ---")
        logger.info("Model output (%s): %s", last.type, last.content)
        return None
    
    def after_agent(self, state, runtime):
        logger.info("=== AGENT END ===")
        logger.info("Final messages:")
        for m in state["messages"]:
            logger.info("%s: %s", m.type, m.content)
        return None

    # Around every tool call
    async def awrap_tool_call(self, request, handler):
        name = request.tool_call["name"]
        args = request.tool_call["args"]
        logger.info(">>> TOOL CALL (async): %s(%s)", name, args)
        try:
            result = await handler(request)
            logger.info("<<< TOOL RESULT (async) from %s: %s", name, getattr(result, "content", result))
            return result
        except Exception as e:
            logger.exception("!!! TOOL ERROR (async) in %s: %s", name, e)
            raise
