from langgraph.graph import StateGraph, START, END 
from src.states.queryState import SparrowAgentState, SparrowInputState
from src.nodes.queryNode import clarify_with_user, write_query_brief

graph_builder = StateGraph(SparrowAgentState, input_schema=SparrowInputState)

graph_builder.add_node("clarify_with_user", clarify_with_user)
graph_builder.add_node("write_query_brief", write_query_brief)

graph_builder.add_edge(START, "clarify_with_user")
graph_builder.add_edge("write_query_brief", END)

query_graph=graph_builder.compile()