from datetime import datetime
from typing_extensions import Literal

from src.llms.groqllm import GroqLLM
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, get_buffer_string
from langgraph.types import Command

from src.utils.prompts import clarification_with_user_instructions, transform_messages_into_customer_query_brief_prompt
from src.states.queryState import SparrowAgentState, ClarifyWithUser, CustomerQuestion

from src.utils.utils import get_today_str

model = GroqLLM().get_llm()

def clarify_with_user(state:SparrowAgentState) -> Command[Literal["write_query_brief", "__end__"]]:
    """
    Determine if the user's request contains sufficient information to proceed with customer request.

    Uses structured output to make deterministic decisions and avoid hallucinations.
    Routes to either customer query brief generation or ends with clarification question.

    """

    structured_output_model = model.with_structured_output(ClarifyWithUser)

   
    response = structured_output_model.invoke([
        SystemMessage(
                content="Route the input to yes or no based on the need of clarification of the query"
            ),
        HumanMessage(
            content=clarification_with_user_instructions.format(
                messages=get_buffer_string(messages=state["messages"]),
                date=get_today_str()
            )
        )
        ], 
    )
    print("RESPONSE", response)

    if response.need_clarification == 'yes':
        return Command(
            goto="need_clarification",
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_query_brief",
            update={"messages": [AIMessage(content=response.verification)]}
        )
    
def write_query_brief(state: SparrowAgentState):
    """
    Transform the conversation history into a comprehensive customer query brief.

    Uses structured output to ensure the brief follows the required format 
    and contains all necessary details for effective research.
    """
    structured_output_model = model.with_structured_output(CustomerQuestion)

    messages = state.get("messages", [])
    print("STATE MESSAGES:", state.get("messages", []))

    if not messages:
        print("ERROR: No messages in state")
        return {
            "query_brief": "",
            "master_messages": []
        }
    
    prompt = transform_messages_into_customer_query_brief_prompt.format(
        messages=get_buffer_string(messages),
        date=get_today_str()
    )
    print("PROMPT", prompt)

    raw_response = model.invoke([HumanMessage(content=prompt)])
    print("RAW MODEL RESPONSE:", raw_response)

    response = structured_output_model.invoke([HumanMessage(content=prompt)])
    print("STRUCTURED RESPONSE:", response)

    if response is None:
        print("ERROR: Structured response is None")
        return {
            "query_brief": "",
            "master_messages": []
        }

    
    return {
        "query_brief": response.query_brief,
        "master_messages": [HumanMessage(content=response.query_brief)]
    }
    
