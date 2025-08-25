from typing_extensions import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class MasterState(TypedDict):
    master_messages: Annotated[Sequence[BaseMessage], add_messages]
    job_type: str  
    sub_agent_output: str
