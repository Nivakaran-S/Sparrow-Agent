from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, filter_messages
from src.utils.prompts import execution_agent_prompt, compress_execution_system_prompt, compress_execution_human_message
from src.utils.utils import think_tool, track_package, get_user_information, estimated_time_analysis

tools = [think_tool, track_package, get_user_information, estimated_time_analysis]
tools_by_name = {tool.name: tool for tool in tools}


class ExecutorNode:
    """
    Executor node for handling tasks:
    1. LLM reasoning
    2. Tool invocation
    3. Final compression
    """

    def __init__(self, llm):
        self.llm = llm
        self.model_with_tools = llm.bind_tools(tools)
        self.MAX_ITERATIONS = 5

    def llm_call(self, state: dict) -> dict:
        """
        Calls the LLM with the executor message history and returns updated state.
        """
        messages = [SystemMessage(content=execution_agent_prompt)] + state.get("executor_messages", [])
        response = self.model_with_tools.invoke(messages)

        return {
            **state,
            "executor_messages": state.get("executor_messages", []) + [response]
        }

    def tool_node(self, state: dict) -> dict:
        """
        Executes any tools requested by the LLM and appends ToolMessages.
        """
        last_message = state.get("executor_messages", [])[-1] if state.get("executor_messages") else None
        tool_calls = getattr(last_message, "tool_calls", []) if last_message else []

        tool_outputs, new_data = [], []
        for call in tool_calls:
            tool_name, args, tool_id = call.get("name"), call.get("args", {}), call.get("id")
            if tool_name in tools_by_name:
                try:
                    result = tools_by_name[tool_name].invoke(args)
                    tool_outputs.append(ToolMessage(content=str(result), name=tool_name, tool_call_id=tool_id))
                    new_data.append(str(result))
                except Exception as e:
                    error_msg = f"Tool {tool_name} failed: {e}"
                    tool_outputs.append(ToolMessage(content=error_msg, name=tool_name, tool_call_id=tool_id))
                    new_data.append(error_msg)

        return {
            **state,
            "executor_messages": state.get("executor_messages", []) + tool_outputs,
            "executor_data": state.get("executor_data", []) + new_data
        }

    def compress_execution(self, state: dict) -> dict:
        """
        Summarizes the execution and returns final structured output.
        """
        messages = [
            SystemMessage(content=compress_execution_system_prompt),
            *state.get("executor_messages", []),
            HumanMessage(content=compress_execution_human_message.format(shipment_request="Track parcel ABC123"))
        ]

        response = self.llm.invoke(messages)

        executor_data = [
            str(m.content)
            for m in filter_messages(state.get("executor_messages", []), include_types=["tool", "ai"])
        ]

        return {
            "output": str(response.content),
            "executor_data": executor_data,
            "executor_messages": state.get("executor_messages", [])
        }

    def route_after_llm(self, state: dict) -> str:
        """Route: decide whether to call a tool or finalize."""
        last_msg = state["executor_messages"][-1]
        return "tool_node" if getattr(last_msg, "tool_calls", None) else "compress_execution"

    def guard_llm(self, state: dict) -> str:
        """
        Prevent infinite loops by limiting iterations.
        """
        iteration_count = state.get("iteration_count", 0) + 1
        state["iteration_count"] = iteration_count
        if iteration_count > self.MAX_ITERATIONS:
            return "compress_execution"
        return self.route_after_llm(state)
