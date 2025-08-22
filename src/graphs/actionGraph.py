from langgraph.graph import StateGraph, START, END
from src.states.actionState import ExecutorState
from src.nodes.actionNode import llm_call, tool_node, compress_execution

agent_builder = StateGraph(ExecutorState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_execution", compress_execution)

# Add edges 
agent_builder.add_edge(START, "llm_call")
agent_builder.add_edge("llm_call", "tool_node")

agent_builder.add_edge("tool_node", "compress_execution")
agent_builder.add_edge("compress_execution", END)

graph = agent_builder.compile()

