from typing import Literal
from langgraph.graph import StateGraph, MessagesState, END

from ragu.utils.state import AgentState, OutputState
from ragu.utils.tool_call import call_tool
#from ragu.utils.classification_agent import openai_inference_generate
from ragu.utils.retrieval_agent import openai_inference_scrape

def route_toolcall(state: AgentState) -> Literal["Action", END]:
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if last_message.tool_calls:
        return "Action"
    # Otherwise if there is, we continue
    else:
        return END

graph = StateGraph(AgentState, input=MessagesState, output=OutputState)
graph.add_node("Retrieval", openai_inference_scrape)
graph.add_node("Action", call_tool)
graph.add_conditional_edges("Retrieval", route_toolcall)
graph.add_edge("Action", "Retrieval")
graph.add_edge("Retrieval", END)
graph.set_entry_point("Retrieval")
graph = graph.compile()
