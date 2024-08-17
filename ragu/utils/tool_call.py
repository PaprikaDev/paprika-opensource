from ragu.utils.tools import tool_node

# Run a tool ordered by the model
def call_tool(state):
    # Define the tool calls
    tool_calls = state['messages'][-1].tool_calls
    # Call the tools and return
    return tool_node.invoke({"messages": [tool_calls]})