from ragu.utils.state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.messages import RemoveMessage
from tools import tools

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
)

model.bind_tools(tools)

retrieval_prompt = """
    You are tasked with searching for a given restaurant menu, downloading the menu, and the upserting the menu to a vector database. 
    ----------
    You will first search for the restaurant menu, you will either download the PDF file or scrape the text directly from the page if a PDF is not available.
    ONLY scrape menu data, this includes foods that the restaurant offers and DOES NOT include information about the restaurant itself or the menu iteself.
    Use the search tool to find the url of the menu, if you cannot find the pdf url directly use the scrape_pdf tool to scrape webpages for the url.
    ---------
    Once you have downloaded or scraped the menu, you will then upsert it into our vector database. 
    You will provide links to the pdf_file holding the menu, aswell as the name of the restaurant and the location of the restaurant.
    ---------
"""

def openai_inference_scrape(state: AgentState):
    messages = [
        {"role": "system", "context": retrieval_prompt}
    ] + state['messages']

    response = model.invoke(messages)
    if len(response.tool_calls) == 0:
        return {'messages': [response]}
    else:
        requirements = response.tool_calls[0]['args']['requirements']
        delete_messages = [RemoveMessage(id=m.id) for m in state['messages']]
        return {"requirements": requirements, "messages": delete_messages}
