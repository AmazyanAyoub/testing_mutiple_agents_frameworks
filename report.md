# Agent Frameworks (Simple Report)

This document explains the best free agent frameworks today.
Also we explain where **pydanticAI** fits.

The goal is to understand:
- which ones are true “agent frameworks”
- which ones are schema / structure helpers
- which ones support Ollama, Bedrock, MCP

---

## 1) CrewAI

- Real agent framework
- Multi-agents that can talk and work together
- Good for production workflows
- Supports Bedrock
- Supports Ollama (via adapters)
- Supports MCP
- Very good documentation and examples online

**When to use it:**  
When you want many agents working together in real tasks.

---

## 2) Microsoft AutoGen

- Real agent framework
- Agents can talk to each other (very clear design)
- Very strong tools system
- Supports Ollama
- Supports Bedrock
- Supports MCP (official adapter)
- Very big documentation (Microsoft Research level)

**When to use it:**  
When you want clean multi-agent chat / hand-off logic.

---

## 3) LlamaIndex Agents

- Real agent framework
- Very strong for data + retrieval + knowledge bases
- Supports Ollama directly
- Supports Bedrock directly
- Supports MCP both ways (client and server)
- Huge documentation

**When to use it:**  
When your agents need strong retrieval and document reasoning.

---

## 4) Haystack Agents

- Real agent framework
- Pipeline style: steps + agent hops
- Supports Ollama
- Supports Bedrock
- Supports MCP (has “mcp-haystack” plugin)
- Good docs

**When to use it:**  
When you want more deterministic pipeline logic with agents.

---

## 5) Phidata (Agno)

- Light weight agent runtime
- Easy UI / dashboards
- Supports Ollama
- Supports Bedrock
- Supports MCP
- Good docs and tutorials

**When to use it:**  
When you want fast, simple agents with UI and tool integration.

---

## Where does **pydanticAI** fit?

**pydanticAI is NOT a full agent framework.**

It is a:
- “LLM + function + validation” tool
- Best for forcing the model to return clean structured data
- Not made for multi-agents
- No orchestration / graph / planner built-in

So:

| Topic | pydanticAI | CrewAI / AutoGen / etc |
|---|---|---|
| Tools / Functions | ✅ strong | ✅ |
| Structured JSON output | ✅ elite | ⚠️ not always strict |
| Multi Agents | ❌ | ✅ |
| Memory / Graph | manual | built-in or easier |

**Perfect use of pydanticAI:**
Use it inside your agents to validate a tool output before using it.

Example idea:
LangGraph node → calls pydanticAI function → validates schema → next node

---

## Final Summary

- Real agent frameworks (full orchestration):
  - CrewAI
  - AutoGen
  - LlamaIndex Agents
  - Haystack Agents
  - Phidata (Agno)

- Helper framework for strict output:
  - pydanticAI (not full agents)

If you are already using LangChain + LangGraph,
you can still add pydanticAI inside them for 100% clean outputs.

---
