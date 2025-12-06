from typing import List, Annotated, Literal
from typing_extensions import TypedDict
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    retrieved_documents: List[str]
    original_query: str
    retrieval_query: str
    input_language: Literal["en", "hi", "hinglish"]
    final_answer: str
    next_step: Literal["translator", "router", "refine_query", "generate_answer"]