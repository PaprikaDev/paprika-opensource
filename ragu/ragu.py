from langgraph.graph import StateGraph, END
from ragu.utils.state import AgentState
from ragu.utils.nodes import openai_inference_scrape, openai_inference_generate, call_tool, requires_tool, exists_record
from ragu.utils.tools import tools

graph = StateGraph(AgentState)

graph.add_node("scrape_llm", openai_inference_scrape)
graph.add_node("generate_llm", openai_inference_generate)
graph.add_node("action", call_tool)

# If the record already exists move to Augmentation + Generation
# Else scrape the record with the Retrieval agent
graph.add_conditional_edges(
    "scrape_llm",
    exists_record,
    {True: "generate_llm", False: "scrape_llm"}
)
graph.add_conditional_edges(
    "scrape_llm",
    requires_tool,
    {True: "action", False: "generate_llm"}
)

graph.add_edge("action", "scrape_llm")
graph.add_edge("generate_llm", END)
graph.set_entry_point("scrape_llm")

graph = graph.compile()