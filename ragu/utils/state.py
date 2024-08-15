from langgraph.graph import add_messages, MessagesState
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence

class AgentState(MessagesState):
    requirements: str
    reqs_gathered: bool

class OutputState(TypedDict):
    response: str