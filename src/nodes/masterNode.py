from langchain_core.messages import SystemMessage, HumanMessage
from src.llms.groqllm import GroqLLM
from src.states.masterState import MasterState, ExecutorState
from src.nodes.actionNode import ExecutorNode
import asyncio
from typing import List, Dict
from src.utils.prompts import master_agent_prompt

class MasterNode:
    def __init__(self, llm: GroqLLM):
        self.llm = llm
        self.executor_node = ExecutorNode(llm)  # Reuse ExecutorNode instance

    async def master_llm_node(self, state: MasterState) -> MasterState:
        if 'query_brief' not in state or not isinstance(state['query_brief'], str):
            state['sub_tasks'] = []
            state['error'] = "Invalid or missing query_brief in state"
            return state

        messages = [
            SystemMessage(content=master_agent_prompt, max_concurrent_logistics_units=3),
            HumanMessage(content=f"Query brief: {state['query_brief']}")
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            sub_tasks = [task.strip() for task in response.content.split(",") if task.strip() and self._is_valid_task(task.strip())]
            state['sub_tasks'] = ['track_package']
            print(f"Sub-tasks: {sub_tasks}")
        except (ValueError, TypeError, RuntimeError) as e:
            state['sub_tasks'] = []
            state['error'] = f"LLM invocation failed: {str(e)}"
        return state

    def _is_valid_task(self, task: str) -> bool:
        # Basic validation: non-empty, reasonable length, no special characters
        return len(task) > 0 and all(c.isalnum() or c.isspace() or c in ".,!?" for c in task)

    async def delegate_to_workers(self, state: MasterState) -> MasterState:
        if not state.get('sub_tasks'):
            state['sub_agent_output'] = "No valid sub-tasks generated."
            return state

        async def process_task(task: str) -> Dict:
            executor_state = {
                "executor_messages": [],
                "execution_job": task,
                "executor_data": []
            }
            if not all(key in executor_state for key in ["executor_messages", "execution_job", "executor_data"]):
                return {"output": "Invalid executor state"}
            
            try:
                executor_state = await self.executor_node.llm_call(executor_state)
                executor_state = await self.executor_node.tool_node(executor_state)
                return self.executor_node.compress_execution(executor_state)
            except (ValueError, TypeError, RuntimeError) as e:
                return {"output": f"Task failed: {str(e)}"}

        tasks = [process_task(task) for task in state['sub_tasks']]
        worker_outputs = await asyncio.gather(*tasks, return_exceptions=True)

        valid_outputs = [
            output.get("output", "") if isinstance(output, dict) else f"Task failed: {str(output)}"
            for output in worker_outputs
        ]
        state['sub_agent_output'] = "\n".join(valid_outputs)
        print('Sub Agent Output', state['sub_agent_output'])
        return state

    async def complete_execution(self, state: MasterState) -> MasterState:
        if not state.get('sub_agent_output') or not isinstance(state['sub_agent_output'], str):
            state['sub_agent_output'] = "No valid output generated."
        return state