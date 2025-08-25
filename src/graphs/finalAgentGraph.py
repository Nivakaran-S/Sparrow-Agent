import asyncio
from src.graphs.masterGraph import master_graph
from src.nodes.queryNode import QueryNode
from src.llms.groqllm import GroqLLM
from src.states.queryState import SparrowAgentState, SparrowInputState
from langgraph.graph import StateGraph, START, END
from src.states.masterState import MasterState

llm = GroqLLM().get_llm()
queryNode = QueryNode(llm)
clarify_with_user = queryNode.clarify_with_user
write_query_brief = queryNode.write_query_brief

def run_master_subgraph(state: SparrowAgentState) -> SparrowAgentState:
    master_input = MasterState.from_sparrow(state)

    # Run the master graph
    master_result = master_graph.invoke(master_input)

    # Map MasterState output back to SparrowAgentState
    updated_state = state.update_from_master(master_result)
    return updated_state

def convert_sparrow_to_master(state: SparrowAgentState) -> MasterState:
    return MasterState(
        query_brief=state.get("query_brief", ""),
        sub_tasks=[],
        error=None,
        sub_agent_output=None
    )

def update_sparrow_from_master(sparrow_state: SparrowAgentState, master_state: MasterState) -> SparrowAgentState:
    sparrow_state["master_result"] = master_state.get("sub_agent_output", "")
    return sparrow_state


async def run_master_subgraph_async(state: SparrowAgentState) -> SparrowAgentState:
    master_input = convert_sparrow_to_master(state)
    master_result = await master_graph.ainvoke(master_input)
    return update_sparrow_from_master(state, master_result)

def run_master_subgraph(state: SparrowAgentState) -> SparrowAgentState:
    return asyncio.run(run_master_subgraph_async(state))

sparrowAgentBuilder= StateGraph(SparrowAgentState, input_schema=SparrowInputState)

sparrowAgentBuilder.add_node("clarify_with_user", clarify_with_user)
sparrowAgentBuilder.add_node("write_query_brief", write_query_brief)
sparrowAgentBuilder.add_node("master_subgraph", run_master_subgraph )

# Adding edges
sparrowAgentBuilder.add_edge(START, "clarify_with_user")
sparrowAgentBuilder.add_edge('clarify_with_user', 'write_query_brief')
sparrowAgentBuilder.add_edge("write_query_brief", "master_subgraph")
sparrowAgentBuilder.add_edge("master_subgraph", END)

sparrowAgent = sparrowAgentBuilder.compile()