
from pydantic import BaseModel, Field
from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END 
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from src.llms.groqllm import GroqLLM

from src.states.actionState import ExecutorState, ExecutorOutputState
from src.utils.utils import get_today_str, think_tool, track_package, get_user_information, estimated_time_analysis
from src.utils.prompts import execution_agent_prompt, compress_execution_system_prompt, compress_execution_human_message

tools = [think_tool, track_package, get_user_information, estimated_time_analysis]  
tools_by_name = {tool.name: tool for tool in tools}
model = GroqLLM().get_llm()
model_with_tools = model.bind_tools(tools)


## Agent Nodes
def llm_call(state: ExecutorState) -> ExecutorState:
    """
    Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response.
    """
    messages = [SystemMessage(content=execution_agent_prompt)] + state.get("executor_messages", [])
    response = model_with_tools.invoke(messages, config={"tools": tools})

    return {
        "executor_messages": state.get("executor_messages", []) + [response]
    }

def tool_node(state: ExecutorState) -> ExecutorState:
    """
    Execute all tool calls from the previous LLM responses.
    Returns updated state with tool execution results.
    """
    last_message = state.get("executor_messages", [])[-1] if state.get("executor_messages") else None
    tool_calls = getattr(last_message, "tool_calls", []) if last_message else []

    tool_outputs = []
    for tool_call in tool_calls:
        tool_name = tool_call.get("name")
        args = tool_call.get("args", {})
        tool_id = tool_call.get("id")
        if tool_name in tools_by_name:
            try:
                result = tools_by_name[tool_name].invoke(args)
                tool_outputs.append(ToolMessage(
                    content=str(result),
                    name=tool_name,
                    tool_call_id=tool_id
                ))
            except Exception as e:
                tool_outputs.append(ToolMessage(
                    content=f"Tool {tool_name} failed: {str(e)}",
                    name=tool_name,
                    tool_call_id=tool_id
                ))
    
    return {
        "executor_messages": state.get("executor_messages", []) + tool_outputs
    }

def compress_execution(state:ExecutorState) -> dict:
    """
    Compress execution findings into a concise summary

    Tasks all the execution messages and tool outputs and creates a compressed summary suitable for the master's decision-making.
    """

    system_message = compress_execution_system_prompt
    print(state['executor_messages'])

    messages = [
        SystemMessage(content=system_message),
        *state.get("executor_messages", []),
        HumanMessage(content=compress_execution_human_message.format(shipment_request="Track parcel ABC123"))
    ]

    # Use model_with_tools if you need tool calls
    response = model_with_tools.invoke(messages, config={"tools": tools})

    executor_data = [
        str(m.content) for m in filter_messages(
            state.get("executor_messages", []),
            include_types=["tools", "ai"]
        )
    ]

    return {
        "output": str(response.content),
        "executor_data": "\n".join(executor_data)
    }

