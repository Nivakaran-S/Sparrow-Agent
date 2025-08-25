from pathlib import Path 
from datetime import datetime
from typing_extensions import Annotated, List, Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool, InjectedToolArg

from src.llms.groqllm import GroqLLM

def get_today_str() -> str:
    """Get current data in a human-readable format."""
    return datetime.now().strftime("%a %b %d, %Y")

groq = GroqLLM()

@tool 
def think_tool(reflection:str) -> str:
    """
    Tool for strategic reflection on execution progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in customer query execution workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing execution gaps: What specific execution am I still missing?
    - Before concluding execution: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial execution or information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue execution or provide my output?

    Args: 
        reflection: Your detailed reflection on the execution progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"


@tool(description="Track parcel based on tracking number")
def track_package(tracking_number: str):
    """
    Tool for tracking customer packages/parcels.

    Use this tool to fetch information about the parcel using the
    tracking number provided by the user.

    Args:
      tracking_number(str): the tracking number
    
    Returns:
      A string describing information on the parcel status
    """

    dummy_db = {
        "ABC123": "Parcel is in transit, expected delivery tomorrow.",
        "XYZ999": "Parcel delivered at 2 PM today.",
    }
    return dummy_db.get(tracking_number, "Tracking ID not found.")

@tool(description="Retrieve user information based on their user ID.")
def get_user_information(userId: str) -> str:
    """
    Retrieve user information based on their user ID.

    Args:
        userId: The unique identifier of the user.

    Returns:
        A string containing user details, including their name and parcel delivery history.
    """

    print("get user information tool called")
    return f"This user id {userId} belongs to Nivakaran. He has sent 200 parcels so far."


@tool(description="Estimate delivery time for a parcel based on origin and destination.")
def estimated_time_analysis(destination: str, origin: str) -> str:
    """
    Estimate delivery time for a parcel based on origin and destination.

    Args:
        destination: The destination of the parcel.
        origin: The place from where the parcel delivery begins.

    Returns:
        A string describing the estimated time of delivery.
    """
    print("estimate time analysis tool called")
    return f"Estimated time analysis for the parcel delivery from {origin} to {destination} is 2 days 50 minutes"

@tool
def conduct_execution(execution_jobs: str) -> str:
    """
    Tool for delegating an execution task to a specialized sub-agent.
    """
    return f"Delegated execution job: {execution_jobs}"


@tool
def execution_complete() -> str:
    """
    Tool for indicating the execution process is complete.
    """
    return "Execution complete."