from langchain_core.messages import SystemMessage, HumanMessage
from src.llms.groqllm import GroqLLM
from src.states.masterState import MasterState, ExecutorState
from src.nodes.actionNode import ExecutorNode
import asyncio
from typing import List, Dict


class MasterNode:
    def __init__(self, llm: GroqLLM):
        self.llm = llm

    async def master_llm_node(self, state: MasterState) -> MasterState:
        messages = [
            SystemMessage(content="You are the master agent. Analyze the query and split it into the minimum number of necessary sub-tasks for efficient parallel execution."),
            HumanMessage(content=f"Query brief: {state.get('query_brief', '')}")
        ]
        try:
            response = await self.llm.ainvoke(messages)
            sub_tasks = [task.strip() for task in response.content.split(",") if task.strip()]
            state['sub_tasks'] = sub_tasks
            print(sub_tasks)  # print AFTER assignment
        except Exception as e:
            state['sub_tasks'] = []
            state['error'] = f"LLM invocation failed: {str(e)}"
        return state

    async def delegate_to_workers(self, state: MasterState) -> MasterState:
        if not state.get('sub_tasks'):
            state['sub_agent_output'] = "No valid sub-tasks generated."
            return state

        async def process_task(task: str) -> Dict:
            executor_node = ExecutorNode(self.llm)
            executor_state = {
                "executor_messages": [],
                "execution_job": task,
                "executor_data": []
            }
            try:
                executor_state = await executor_node.llm_call(executor_state)
                executor_state = await executor_node.tool_node(executor_state)
                return executor_node.compress_execution(executor_state)
            except Exception as e:
                return {"output": f"Task failed: {str(e)}"}

        tasks = [process_task(task) for task in state['sub_tasks']]
        worker_outputs = await asyncio.gather(*tasks, return_exceptions=True)

        valid_outputs = [
            output.get("output", "") if isinstance(output, dict) else f"Task failed: {str(output)}"
            for output in worker_outputs
        ]
        state['sub_agent_output'] = "\n".join(valid_outputs)
        return state

    async def complete_execution(self, state: MasterState) -> MasterState:
        state['sub_agent_output'] = state.get('sub_agent_output', "No output generated.")
        return state
