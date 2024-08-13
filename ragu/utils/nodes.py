from state import AgentState
from langchain_core.messages import SystemMessage, ToolMessage

def exists_record(self, state: AgentState):
        # if the current menu is already in the vector database
        # move to the augmentation step 
        return False

# Check if llm requires action 
def requires_tool(self, state: AgentState):
    result = state['messages'][-1]
    return len(result.tool_calls) > 0
    
# Run a tool ordered by the model
def call_tool(self, state: AgentState):
    tool_calls = state['messages'][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling: {t}")
        result = self.tools[t['name']].invoke(t['args'])
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
    print("Back to the model!")
    return {'messages': results}

# Invokes the current message chain
def openai_inference_scrape(self, state: AgentState):
    messages = state['messages']
    if self.R_prompt:
        messages = [SystemMessage(content=self.R_prompt)] + messages
    message = self.model.invoke(messages)
    return {'messages': [message]}

def openai_inference_generate(self, state: AgentState):
    messages = state['messages']
    if self.G_prompt:
        messages = [SystemMessage(content=self.G_prompt)] + messages
    message = self.model.invoke(messages)
    return {'messages': [message]}
