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
