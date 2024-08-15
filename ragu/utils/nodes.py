from langchain_core.messages import SystemMessage, ToolMessage
from ragu.utils.tools import tools
from langgraph.prebuilt import ToolNode

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