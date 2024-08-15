
from ragu.utils.state import AgentState
from langchain_openai import ChatOpenAI
from langchain_core.messages import RemoveMessage
from tools import tools

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

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
)

model.bind_tools(tools)

def exists_record(state):
        # Check if the menu has been gathered yet
        if state['reqs_gathered']:
                return True
        return False

def GetMenu():
       pass

def GetDietTypes():
       # We will eventually pass data from the application itself
       diets = ["Vegan", "Vegetarian"]
       return diets 

def openai_inference_generate(self, state: AgentState):
    menu = GetMenu()
    diet_types = GetDietTypes()
    
    messages = [
        {"role": "system", "context": generation_prompt.format(MENU=menu, DIETTYPE=diet_types)}
    ] + state['messages']
    
    response = model.invoke(messages)
    return {'messages': [response]}
