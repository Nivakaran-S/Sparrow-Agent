from src.graphs.masterGraph import master_graph
from src.nodes.queryNode import QueryNode
from src.llms.groqllm import GroqLLM
from src.states.queryState import SparrowAgentState, SparrowInputState
from langgraph.graph import StateGraph, START, END

llm = GroqLLM().get_llm()
queryNode = QueryNode(llm)
clarify_with_user = queryNode.clarify_with_user
write_query_brief = queryNode.write_query_brief

sparrowAgentBuilder= StateGraph(SparrowAgentState, input_schema=SparrowInputState)

sparrowAgentBuilder.add_node("clarify_with_user", clarify_with_user)
sparrowAgentBuilder.add_node("write_query_brief", write_query_brief)
sparrowAgentBuilder.add_node("master_subgraph", master_graph )

# Adding edges
sparrowAgentBuilder.add_edge(START, "clarify_with_user")
sparrowAgentBuilder.add_edge("write_query_brief", "master_subgraph")
sparrowAgentBuilder.add_edge("master_subgraph", END)

sparrowAgent = sparrowAgentBuilder.compile()