from ragu.utils.state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage
from ragu.utils.tools import tools
from langgraph.prebuilt import ToolNode

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

generation_prompt = """
You are tasked with classifying if a resturant's menu fits a user's diet types. 
----------
MENU:
{MENU}
---------
Diet Types:
{DIETTYPE}
---------
"""

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
)

# def exists_record(state):
#         # if the current menu is already in the vector database
#         # move to the augmentation step 
#         return False

# Check if llm requires action 
def requires_tool(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there are no tool calls, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"
    
# Run a tool ordered by the model
def call_tool(state):
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling: {t}")
        result = tools[t['name']].invoke(t['args'])
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    print("Back to the model!")
    return {'messages': results}

# Invokes the current message chain
def openai_inference_scrape(state):
    messages = state['messages']
    messages = [SystemMessage(content=retrieval_prompt)] + messages
    response = llm.invoke(messages)
    return {'messages': [response]}

# def openai_inference_generate(self, state: AgentState):
#     messages = state['messages']
#     if self.G_prompt:
#         messages = [SystemMessage(content=self.G_prompt)] + messages
#     message = self.model.invoke(messages)
#     return {'messages': [message]}

# Define the function to execute tools
tool_node = ToolNode(tools)