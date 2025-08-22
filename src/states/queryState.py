import operator 
from typing_extensions import Optional, Annotated, List, Sequence, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field 


class SparrowInputState(MessagesState):
    """Input state for the full agent - only contains from the user input."""
    pass 

class SparrowAgentState(MessagesState):
    """
    Main state for the full multi-agent Sparrow customer service system.

    Extends MessagesState with additional fields for Sparrow customer service coordination.
    Note: Some fields are duplicated across different state classes for proper
    state management between subgraphs and main workflow.

    """
    query_brief: Optional[str]
    master_messages: Annotated[Sequence[BaseMessage], add_messages]
    notes: Annotated[list[str], operator.add] = []
    final_message: str

class ClarifyWithUser(BaseModel):
    """Schema for user clarification decision and questions"""

    need_clarification: Literal["yes", "no"] = Field(
        description="Whether the user needs to be asked a clarifying question"
    )
    question: str = Field(
        description="A question to ask the user to clarify the need"
    )
    verification:str = Field(
        description="Verify message that we will start research after the user has provided the necessary information"
    )

class CustomerQuestion(BaseModel):
    """Schema for structured customer query brief """

    query_brief: str = Field(
        description="A customer question that will be used to guide the research."
    )