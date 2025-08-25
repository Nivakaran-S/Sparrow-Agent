from typing_extensions import TypedDict, Annotated, Sequence, List
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_core.tools import tool 
from pydantic import BaseModel, Field

class MasterState(TypedDict):
    master_messages: Annotated[Sequence[BaseMessage], add_messages]
    job_type: str  
    sub_agent_output: str
    query_brief: str


class ExecutorState(BaseModel):
    """
    State for the executor agent containing message history and research metadata.
    """
    executor_messages: Annotated[List[BaseMessage], add_messages]
    execution_job: str
    executor_data: List[str]

@tool 
class ConductExecution(BaseModel):
    """Tool for delegating an execution task to a specialized sub-agent"""
    execution_jobs: str = Field("The jobs to be executed")

@tool 
class ExecutionComplete(BaseModel):
    """Tool for indicating the execution process is complete"""
    pass 





