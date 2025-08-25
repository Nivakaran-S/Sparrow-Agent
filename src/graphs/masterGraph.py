from langgraph.graph import StateGraph, START, END
from src.nodes.masterNode import run_sub_agent, route_job, master_compress
from src.states.masterState import MasterState

master_builder = StateGraph(MasterState)

# Nodes
master_builder.add_node("route_job", route_job)
master_builder.add_node("run_sub_agent", run_sub_agent)
master_builder.add_node("master_compress", master_compress)

# Edges
master_builder.add_edge(START, "route_job")
master_builder.add_edge("route_job", "run_sub_agent")
master_builder.add_edge("run_sub_agent", "master_compress")
master_builder.add_edge("master_compress", END)

master_graph = master_builder.compile()
