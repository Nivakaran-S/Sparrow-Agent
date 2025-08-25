
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


## Agent Node
class ExecutorNode:
    """
    A class to represent executor node

    """
    def __init__(self, llm):
        self.llm = llm 
        self.model_with_tools = llm.bind_tools(tools)
        self.MAX_ITERATIONS = 5

    def llm_call(self, state: ExecutorState) -> ExecutorState:
        """
        Main agent LLM call.

        Prepends a system prompt and feeds in the executor message history.
        Returns the updated state with the LLM's output message.
        """
        messages = [SystemMessage(content=execution_agent_prompt)] + state.get("executor_messages", [])
        response = self.model_with_tools.invoke(messages, config={"tools": tools})

        return {
            "executor_messages": state.get("executor_messages", []) + [response],
            "execution_job": state.get("execution_job", ""),   # preserve context
            "executor_data": state.get("executor_data", []),   # preserve collected data
        }


    ## Tool Execution Node
    def tool_node(self, state: ExecutorState) -> ExecutorState:
        """
        Execute any tool calls requested by the latest LLM message.
        Appends tool outputs as ToolMessages to the executor_messages,
        and stores results in executor_data.
        """
        last_message = state.get("executor_messages", [])[-1] if state.get("executor_messages") else None
        tool_calls = getattr(last_message, "tool_calls", []) if last_message else []

        tool_outputs = []
        new_data = []
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            args = tool_call.get("args", {})
            tool_id = tool_call.get("id")

            if tool_name in tools_by_name:
                try:
                    result = tools_by_name[tool_name].invoke(args)
                    tool_outputs.append(
                        ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_id)
                    )
                    new_data.append(str(result))
                except Exception as e:
                    error_msg = f"Tool {tool_name} failed: {str(e)}"
                    tool_outputs.append(
                        ToolMessage(content=error_msg, name=tool_name, tool_call_id=tool_id)
                    )
                    new_data.append(error_msg)

        return {
            "executor_messages": state.get("executor_messages", []) + tool_outputs,
            "execution_job": state.get("execution_job", ""),
            "executor_data": state.get("executor_data", []) + new_data,
        }


    ## Compression / Finalization Node
    def compress_execution(self, state: ExecutorState) -> ExecutorOutputState:
        """
        Summarize the execution into a final concise output.

        Produces an ExecutorOutputState that includes:
        - output: final answer
        - executor_data: aggregated tool results
        - executor_messages: full conversation trace
        """
        system_message = compress_execution_system_prompt

        messages = [
            SystemMessage(content=system_message),
            *state.get("executor_messages", []),
            HumanMessage(
                content=compress_execution_human_message.format(
                    shipment_request="Track parcel ABC123"
                )
            )
        ]

        response = self.llm.invoke(messages)   # Final summary, no tools required

        executor_data = [
            str(m.content)
            for m in filter_messages(
                state.get("executor_messages", []),
                include_types=["tool", "ai"]
            )
        ]

        return ExecutorOutputState(
            output=str(response.content),
            executor_data=executor_data,
            executor_messages=state.get("executor_messages", []),
        )

    

    def route_after_llm(self, state: ExecutorState):
        """Decide whether to call a tool or finalize."""
        last_msg = state["executor_messages"][-1]
        if getattr(last_msg, "tool_calls", None):
            return "tool_node"
        return "compress_execution"

    def guard_llm(self, state: ExecutorState):
        """
        Router with iteration safeguard.
        Prevents infinite tool/LLM loops.
        """
        iteration_count = state.get("iteration_count", 0) + 1
        state["iteration_count"] = iteration_count
        if iteration_count > self.MAX_ITERATIONS:
            # Force finalize if too many iterations
            return "compress_execution"
        return self.route_after_llm(state)
