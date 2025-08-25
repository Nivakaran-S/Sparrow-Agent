from langgraph.graph import StateGraph, START, END
from src.nodes.masterNode import MasterNode
from src.states.masterState import MasterState
from src.llms.groqllm import GroqLLM


class MasterBuilder:
    def __init__(self, llm):
        self.llm = llm 
        self.graph = StateGraph(MasterState)
    
    def build_master_graph(self):
        master_node_obj = MasterNode(self.llm)

        self.graph.add_node("master_llm_node", master_node_obj.master_llm_node)
        self.graph.add_node("delegate_workers", master_node_obj.delegate_to_workers)
        self.graph.add_node("execution_complete", master_node_obj.complete_execution)

        self.graph.add_edge(START, "master_llm_node")
        self.graph.add_edge("master_llm_node", "delegate_workers")
        self.graph.add_edge("delegate_workers", "execution_complete")
        self.graph.add_edge("execution_complete", END)

        return self.graph 


# ----------------------------
# 5) INITIALIZE & RUN
# ----------------------------
llm = GroqLLM().get_llm()
graph_builder = MasterBuilder(llm)
master_graph = graph_builder.build_master_graph().compile()
print("Graph created successfully")