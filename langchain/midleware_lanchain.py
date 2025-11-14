from dataclasses import dataclass
from langchain.agents.middleware import (
    AgentMiddleware, 
    AgentState, 
    ToolRetryMiddleware,
    ToolCallLimitMiddleware, 
    ModelCallLimitMiddleware,
    hook_config
)
from langgraph.runtime import Runtime
from typing import Any
from langchain_core.messages import ToolMessage, AIMessage
from tiny_classifier import _hash_text, _classify_conditional_sync


import json
from log import logger


@dataclass
class FullLoggingMiddleware(AgentMiddleware):
    _seen_calls = set()
    _last_user_idx: int = -1             # ADD
    _last_user_text: str = ""            # ADD
    _cls_cache: dict = None              # ADD
    _cls: dict | None = None             # ADD
    _conditional_active: bool = False    # ADD

retry_failed_tools = ToolRetryMiddleware(
    max_retries=3,
    backoff_factor=2.0,
    initial_delay=1.0,
    max_delay=60.0,
    jitter=True,
)

reg_limit = ToolCallLimitMiddleware(
    tool_name="lookup_regulation",
    run_limit=1,           # at most once per invocation
    exit_behavior="end",   # stop the run after this tool is exceeded
)

model_limit = ModelCallLimitMiddleware(run_limit=5, thread_limit=50)

@dataclass
class FullLoggingMiddleware(AgentMiddleware):
    _seen_calls = set()

    # Called once per invocation
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        self._seen_calls.clear()
        self._last_user_idx = -1            # ADD
        self._last_user_text = ""           # ADD
        self._cls_cache = {}                # ADD
        self._cls = None                    # ADD
        self._conditional_active = False    # ADD
        logger.info("=== AGENT START ===")
        logger.info("Initial state: %s", state)
        return None

    # Before every model call
    def before_model(self, state, runtime):
        logger.info("--- BEFORE MODEL ---")
        logger.info("Messages (%d):", len(state["messages"]))
        for m in state["messages"]:
            logger.info("%s: %s", m.type, m.content)

        # ADD — detect a new user turn and classify it
        msgs = state["messages"]
        for i in range(len(msgs) - 1, -1, -1):
            if getattr(msgs[i], "type", "") in ("human", "user"):
                if i != self._last_user_idx:
                    txt = msgs[i].content if isinstance(msgs[i].content, str) else ""
                    self._last_user_idx = i
                    self._last_user_text = txt

                    key = _hash_text(txt)
                    cached = self._cls_cache.get(key)
                    if cached is None:
                        self._cls = _classify_conditional_sync(txt)  # sync, small, cheap
                        self._cls_cache[key] = self._cls
                    else:
                        self._cls = cached

                    self._conditional_active = bool(self._cls and self._cls.get("has_conditional"))
                    logger.info("Conditional(classifier) detected: %s | hints=%s",
                                self._conditional_active,
                                (self._cls or {}).get("tool_hints"))
                break

        return None


    # After every model call
    # def after_model(self, state, runtime):
    #     last = state["messages"][-1]
    #     logger.info("--- AFTER MODEL ---")
    #     logger.info("Model output (%s): %s", last.type, last.content)
    #     return None

    @hook_config(can_jump_to=["model"])  # ADD
    def after_model(self, state, runtime):
        # keep your logs
        last = state["messages"][-1]
        logger.info("--- AFTER MODEL ---")
        logger.info("Model output (%s): %s", last.type, last.content)

        # ====== CONDITIONAL ENFORCEMENT (generic) ======
        if not self._conditional_active or not self._cls:
            return None

        # Messages since last user turn
        msgs = state["messages"]
        since = msgs[self._last_user_idx + 1 :] if self._last_user_idx >= 0 else msgs

        # Which tools were used since last user turn?
        used_tools = {
            (getattr(m, "name", "") or getattr(m, "tool_name", "")).lower()
            for m in since
            if isinstance(m, ToolMessage)
        }

        # Candidate tools to execute the action (from classifier hints)
        action_needs = set(
            (self._cls.get("action", {}) or {}).get("needs_tools", []) or []
        )
        tool_hints = set(self._cls.get("tool_hints", []) or [])
        candidates = {t.lower() for t in (action_needs or tool_hints)}

        # Try to evaluate simple numeric predicates using latest numeric ToolMessage
        pred = self._cls.get("predicate", {}) or {}
        op  = (pred.get("op") or "").strip()
        rhs = pred.get("rhs")
        lhs_value = None

        # Find the most recent numeric tool result (generic)
        for m in reversed(since):
            if isinstance(m, ToolMessage):
                try:
                    # ToolMessage.content may be a str; try to parse a float out of it
                    c = m.content
                    if isinstance(c, (int, float)):
                        lhs_value = float(c)
                        break
                    if isinstance(c, str):
                        lhs_value = float(c.strip())
                        break
                except Exception:
                    continue

        # Evaluate predicate if we have op, rhs (numeric), and a numeric lhs_value
        true_predicate = False
        try:
            if op and rhs is not None and lhs_value is not None:
                r = float(rhs)
                if   op == ">":  true_predicate = lhs_value >  r
                elif op == "<":  true_predicate = lhs_value <  r
                elif op == ">=": true_predicate = lhs_value >= r
                elif op == "<=": true_predicate = lhs_value <= r
                elif op == "==": true_predicate = lhs_value == r
                elif op == "!=": true_predicate = lhs_value != r
                # (text ops like contains can be added later if needed)
        except Exception:
            pass

        # If predicate is TRUE and no relevant tool was used yet, bounce back to model to call one
        if true_predicate and candidates and used_tools.isdisjoint(candidates):
            hint = ", ".join(sorted(list(candidates))[:2]) or "a relevant tool"
            return {
                "messages": [
                    AIMessage(
                        content=(
                            f"Condition evaluated TRUE (value={lhs_value} {op} {rhs}). "
                            f"Call one relevant tool now (e.g., {hint}) to execute the requested action, "
                            f"then provide the final answer."
                        )
                    )
                ],
                "jump_to": "model",
            }

        # If predicate is FALSE and model claims action results without any tool usage, nudge it to state skip explicitly
        if not true_predicate and candidates and not used_tools:
            # Only nudge if it wrote text already (avoid loops when it hasn't answered)
            if getattr(last, "type", "") == "ai" and isinstance(last.content, str) and last.content.strip():
                return {
                    "messages": [
                        AIMessage(
                            content=(
                                "The condition evaluated FALSE based on available results; "
                                "do NOT execute any tools. Respond briefly explaining that the action was skipped."
                            )
                        )
                    ],
                    "jump_to": "model",
                }

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

        key = (name, json.dumps(args, sort_keys=True))
        if key in self._seen_calls:
            return ToolMessage(
                content="Duplicate call suppressed.",
                name=name,
                tool_call_id=request.tool_call.get("id"),
            )
        self._seen_calls.add(key)
        logger.info(">>> TOOL CALL (async): %s(%s)", name, args)
        try:
            result = await handler(request)
            logger.info("<<< TOOL RESULT (async) from %s: %s", name, getattr(result, "content", result))
            return result
        except Exception as e:
            logger.exception("!!! TOOL ERROR (async) in %s: %s", name, e)
            raise
