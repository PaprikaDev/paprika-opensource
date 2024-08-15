from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END, START
from ragu.utils.state import AgentState
from ragu.utils.nodes import openai_inference_scrape, call_tool, requires_tool

graph = StateGraph(AgentState)

class GraphConfig(TypedDict):
    model_name: Literal["openai"]

graph.add_node("scrape_llm", openai_inference_scrape)
# graph.add_node("generate_llm", openai_inference_generate)
graph.add_node("action", call_tool)

graph.add_edge(START, "scrape_llm")

# If the record already exists move to Augmentation + Generation
# Else scrape the record with the Retrieval agent
# graph.add_conditional_edges(
#     "scrape_llm",
#     exists_record,
#     {True: "generate_llm", False: "scrape_llm"}
# )
graph.add_conditional_edges(
    "scrape_llm",
    requires_tool,
    {"continue": "action", "end": END}
)

graph.add_edge("action", "scrape_llm")
# graph.add_edge("generate_llm", END)

graph = graph.compile()

# retrieval_prompt = """
# You are tasked with searching for a given restaurant menu, downloading the menu, and the upserting the menu to a vector database. 
# ----------
# You will first search for the restaurant menu, you will either download the PDF file or scrape the text directly from the page if a PDF is not available.
# ONLY scrape menu data, this includes foods that the restaurant offers and DOES NOT include information about the restaurant itself or the menu iteself.
# Use the search tool to find the url of the menu, if you cannot find the pdf url directly use the scrape_pdf tool to scrape webpages for the url.
# ---------
# Once you have downloaded or scraped the menu, you will then upsert it into our vector database. 
# You will provide links to the pdf_file holding the menu, aswell as the name of the restaurant and the location of the restaurant.
# ---------
# """

# generation_prompt = """
# You are tasked with classifying if a resturant's menu fits a user's diet types. 
# ----------
# MENU:
# {MENU}
# ---------
# Diet Types:
# {DIETTYPE}
# ---------
# """

# llm = ChatOpenAI(
#     model="gpt-4o",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
# )