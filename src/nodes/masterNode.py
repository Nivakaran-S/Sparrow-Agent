from langchain_core.messages import SystemMessage, HumanMessage
from src.llms.groqllm import GroqLLM
from src.states.masterState import MasterState
from src.graphs.actionGraph import graph

router_prompt = """
You are the MASTER AGENT. 
Decide which specialist agent should handle the user request.

Options:
- "track_parcel" → use the Parcel Tracking Agent
- "eta_analysis" → use the ETA Analysis Agent
- "user_info" → use the User Info Agent

Respond ONLY with one job_type.
"""

model = GroqLLM().get_llm()


def route_job(state: MasterState) -> MasterState:
    messages = [SystemMessage(content=router_prompt)] + state["master_messages"]
    response = model.invoke(messages)
    job_type = response.content.strip()
    return {**state, "job_type": job_type}

def run_sub_agent(state: MasterState) -> MasterState:
    job = state["job_type"]
    user_message = state["master_messages"][-1]

    if job == "track_parcel":
        executor_state = {
            "executor_messages": [user_message],
            "execution_job": "track_parcel",
            "executor_data": [],
            "iteration_count": 0,
        }
        result = graph.invoke(executor_state)   # call your existing executor agent

    elif job == "eta_analysis":
        # TODO: call ETA agent graph
        result = {"output": "ETA is approx 2 days", "executor_data": [], "executor_messages": []}

    elif job == "user_info":
        # TODO: call User Info agent graph
        result = {"output": "User is premium member", "executor_data": [], "executor_messages": []}

    else:
        result = {"output": f"No sub-agent found for {job}", "executor_data": [], "executor_messages": []}

    return {**state, "sub_agent_output": result["output"]}

def master_compress(state: MasterState) -> MasterState:
    summary = f"Job: {state['job_type']}\nResult: {state['sub_agent_output']}"
    return {**state, "sub_agent_output": summary}
