from langgraph.graph import StateGraph, START, END
from src.states.actionState import ExecutorState, ExecutorOutputState
from src.nodes.actionNode import llm_call, tool_node, compress_execution

# Max iterations safeguard
MAX_ITERATIONS = 5

def route_after_llm(state: ExecutorState):
    """Decide whether to call a tool or finalize."""
    last_msg = state["executor_messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tool_node"
    return "compress_execution"

def guard_llm(state: ExecutorState):
    """
    Router with iteration safeguard.
    Prevents infinite tool/LLM loops.
    """
    iteration_count = state.get("iteration_count", 0) + 1
    state["iteration_count"] = iteration_count
    if iteration_count > MAX_ITERATIONS:
        # Force finalize if too many iterations
        return "compress_execution"
    return route_after_llm(state)


# -----------------------------
# Graph Construction
# -----------------------------
agent_builder = StateGraph(ExecutorState, output=ExecutorOutputState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_execution", compress_execution)

# Flow
agent_builder.add_edge(START, "llm_call")

agent_builder.add_conditional_edges(
    "llm_call",
    guard_llm,
    {
        "tool_node": "tool_node",
        "compress_execution": "compress_execution",
    },
)

# After tools, loop back into LLM
agent_builder.add_edge("tool_node", "llm_call")

# Exit
agent_builder.add_edge("compress_execution", END)

graph = agent_builder.compile()
