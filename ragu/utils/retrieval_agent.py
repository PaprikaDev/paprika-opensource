from ragu.utils.state import AgentState
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
# from langchain_core.messages import RemoveMessage
from ragu.utils.tools import tools

search_tool = TavilySearchResults()

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    max_tokens=None,
    timeout=None,
)

retrieval_prompt = """
    You are tasked with searching for a given restaurant menu, downloading the menu, and the upserting the menu to a vector database. 
    ----------
    You will first search for the restaurant menu, you will find and download the PDF of the menu.
    After you have download the PDF, you will then upsert it into a vector database. 
    ONLY download menu data, this includes foods that the restaurant offers and DOES NOT include information about the restaurant itself or the menu iteself.
    Use the search tool to find the url of the pdf menu, the menu MUST EXIST and must have been found using the search tool.
    ---------
    Once you have downloaded the menu, you will then upsert it into our vector database using the upsert_pdf tool. 
    You will provide links to the pdf_file holding the menu, aswell as the name of the restaurant and the location of the restaurant.
    ---------
    {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(retrieval_prompt)

agent = create_openai_tools_agent(model, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools)

def openai_inference_scrape(state: AgentState):
    messages = state['messages']

    response = agent_executor.invoke(
        {
            "input": messages[-1],
            "chat_history": messages
        }
    )

    if len(response.tool_calls) == 0:
        return {'messages': [response]}
    else:
        requirements = response.tool_calls[0]['args']['requirements']
        # delete_messages = [RemoveMessage(id=m.id) for m in state['messages']]
        return {"requirements": requirements, "messages": messages + response}
