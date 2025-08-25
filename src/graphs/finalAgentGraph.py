# Updated Sparrow Agent with proper routing
import asyncio
import logging
from src.graphs.masterGraph import master_graph
from src.llms.groqllm import GroqLLM
from src.states.queryState import SparrowAgentState, SparrowInputState
from langgraph.graph import StateGraph, START, END
from src.states.masterState import MasterState
from langgraph.checkpoint.memory import MemorySaver
from src.nodes.queryNode import QueryNode
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

llm = GroqLLM().get_llm()
queryNode = QueryNode(llm)

def convert_sparrow_to_master(state: SparrowAgentState) -> MasterState:
    return MasterState(
        query_brief=state.get("query_brief", ""),
        execution_jobs=[],
        completed_jobs=[],
        worker_outputs=[],
        final_output=''
    )

def update_sparrow_from_master(sparrow_state: SparrowAgentState, master_state: MasterState) -> SparrowAgentState:
    sparrow_state["master_result"] = master_state.get("final_output", "")
    return sparrow_state

def route_after_clarification(state: SparrowAgentState) -> str:
    """Route based on clarification status"""
    
    # Check if we have an error
    if state.get("error"):
        print(f"Error detected: {state.get('error')}")
        return "__end__"
    
    # Check clarification status
    clarification_complete = state.get("clarification_complete", False)
    needs_clarification = state.get("needs_clarification", True)
    
    print(f"Clarification complete: {clarification_complete}")
    print(f"Needs clarification: {needs_clarification}")
    
    if clarification_complete or not needs_clarification:
        return "write_query_brief"
    else:
        return "need_clarification"

def route_after_query_brief(state: SparrowAgentState) -> str:
    """Route after query brief creation"""
    
    # Check for errors
    if state.get("error"):
        print(f"Error in query brief: {state.get('error')}")
        return "__end__"
    
    # Check if query brief is adequate
    query_brief = state.get("query_brief", "")
    query_brief_complete = state.get("query_brief_complete", False)
    
    if query_brief_complete and query_brief and len(query_brief.strip()) > 10:
        return "master_subgraph"
    else:
        print("Query brief insufficient, going back to clarification")
        return "clarify_with_user"

def need_clarification(state: SparrowAgentState) -> SparrowAgentState:
    """Handle case where clarification is needed - wait for user input"""
    print("Waiting for user clarification...")
    # In a real system, this would pause and wait for user input
    # For now, just return the state as-is
    return state

async def run_master_subgraph_async(state: SparrowAgentState) -> SparrowAgentState:
    try:
        print("Running master subgraph...")
        master_input = convert_sparrow_to_master(state)
        master_result = await master_graph.ainvoke(master_input)
        return update_sparrow_from_master(state, master_result)
    except Exception as e:
        logger.error(f"Master subgraph failed: {e}")
        return {**state, "error": str(e)}

def run_master_subgraph(state: SparrowAgentState) -> SparrowAgentState:
    try:
        return asyncio.run(run_master_subgraph_async(state))
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(run_master_subgraph_async(state))
        raise e

# Build the graph
sparrowAgentBuilder = StateGraph(SparrowAgentState, input_schema=SparrowInputState)

sparrowAgentBuilder.add_node("clarify_with_user", queryNode.clarify_with_user)
sparrowAgentBuilder.add_node("need_clarification", need_clarification)
sparrowAgentBuilder.add_node("write_query_brief", queryNode.write_query_brief)
sparrowAgentBuilder.add_node("master_subgraph", run_master_subgraph)

# Edges
sparrowAgentBuilder.add_edge(START, "clarify_with_user")

sparrowAgentBuilder.add_conditional_edges(
    "clarify_with_user",
    route_after_clarification,
    {
        "need_clarification": "need_clarification",
        "write_query_brief": "write_query_brief",
        "__end__": END
    }
)

# From need_clarification, go back to clarify_with_user (after user provides input)
sparrowAgentBuilder.add_edge("need_clarification", END)  # Or create a loop back

sparrowAgentBuilder.add_conditional_edges(
    "write_query_brief",
    route_after_query_brief,
    {
        "clarify_with_user": "clarify_with_user",
        "master_subgraph": "master_subgraph",
        "__end__": END
    }
)

sparrowAgentBuilder.add_edge("master_subgraph", END)


sparrowAgent = sparrowAgentBuilder.compile()
