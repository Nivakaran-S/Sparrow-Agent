from langgraph.graph import StateGraph, START, END
from src.states.actionState import ExecutorState, ExecutorOutputState

from src.nodes.actionNode import ExecutorNode
from src.llms.groqllm import GroqLLM


class ExecutorGraphBuilder:
    def __init__(self, llm):
        self.llm = llm 
        self.graph = StateGraph(ExecutorState, output=ExecutorOutputState)
    
    def build_executor_graph(self):
        """
        Build a graph to build the executor

        """
        self.executor_node_obj = ExecutorNode(self.llm)
        print(self.llm)

        self.graph.add_node("llm_call", self.executor_node_obj.llm_call)
        self.graph.add_node("tool_node", self.executor_node_obj.tool_node)
        self.graph.add_node("compress_execution", self.executor_node_obj.compress_execution)

        # Flow
        self.graph.add_edge(START, "llm_call")

        self.graph.add_conditional_edges(
            "llm_call",
            self.executor_node_obj.guard_llm,
            {
                "tool_node": "tool_node",
                "compress_execution": "compress_execution",
            },
        )

        # After tools, loop back into LLM
        self.graph.add_edge("tool_node", "llm_call")

        # Exit
        self.graph.add_edge("compress_execution", END)

        return self.graph
    
    def setup_graph(self):
        return self.graph.compile()
    



llm=GroqLLM().get_llm()

## get the graph
graph_builder=ExecutorGraphBuilder(llm)
graph=graph_builder.build_executor_graph().compile()
