



from datetime import datetime
from typing_extensions import Literal
from src.llms.groqllm import GroqLLM
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, get_buffer_string
from src.utils.prompts import clarification_with_user_instructions, transform_messages_into_customer_query_brief_prompt
from src.states.queryState import SparrowAgentState, ClarifyWithUser, CustomerQuestion
from src.utils.utils import get_today_str

class QueryNode:
    def __init__(self, llm):
        self.llm = llm
        
    def clarify_with_user(self, state: SparrowAgentState) -> SparrowAgentState:
        """
        Determine if the user's request contains sufficient information to proceed.
        Returns updated state with clarification status.
        """
        structured_output_model = self.llm.with_structured_output(ClarifyWithUser)
        
        try:
            response = structured_output_model.invoke([
                SystemMessage(
                    content="Route the input to yes or no based on the need of clarification of the query"
                ),
                HumanMessage(
                    content=clarification_with_user_instructions.format(
                        messages=get_buffer_string(messages=state.get("messages", [])),
                        date=get_today_str()
                    )
                )
            ])
            
            print("CLARIFICATION RESPONSE:", response)
            
            # Update state based on response
            updated_state = {**state}
            
            if response.need_clarification == 'yes':
                updated_state.update({
                    "messages": state.get("messages", []) + [AIMessage(content=response.question)],
                    "clarification_complete": False,
                    "needs_clarification": True
                })
            else:
                updated_state.update({
                    "messages": state.get("messages", []) + [AIMessage(content=response.verification)],
                    "clarification_complete": True,
                    "needs_clarification": False
                })
            
            return updated_state
            
        except Exception as e:
            print(f"Error in clarify_with_user: {e}")
            return {
                **state,
                "clarification_complete": False,
                "needs_clarification": True,
                "error": str(e)
            }

    def write_query_brief(self, state: SparrowAgentState) -> SparrowAgentState:
        """
        Transform the conversation history into a comprehensive customer query brief.
        """
        try:
            structured_output_model = self.llm.with_structured_output(CustomerQuestion)
            
            messages = state.get("messages", [])
            print("STATE MESSAGES:", messages)
            
            if not messages:
                print("ERROR: No messages in state")
                return {
                    **state,
                    "query_brief": "",
                    "error": "No messages available for query brief creation"
                }
            
            prompt = transform_messages_into_customer_query_brief_prompt.format(
                messages=get_buffer_string(messages),
                date=get_today_str()
            )
            print("PROMPT:", prompt)
            
            # Test raw response first
            raw_response = self.llm.invoke([HumanMessage(content=prompt)])
            print("RAW MODEL RESPONSE:", raw_response)
            
            # Get structured response
            response = structured_output_model.invoke([HumanMessage(content=prompt)])
            print("STRUCTURED RESPONSE:", response)
            
            if response is None:
                print("ERROR: Structured response is None")
                return {
                    **state,
                    "query_brief": "",
                    "error": "Failed to generate structured response"
                }
            
            return {
                **state,
                "query_brief": response.query_brief,
                "master_messages": [HumanMessage(content=response.query_brief)],
                "query_brief_complete": True
            }
            
        except Exception as e:
            print(f"Error in write_query_brief: {e}")
            return {
                **state,
                "query_brief": "",
                "error": str(e)
            }