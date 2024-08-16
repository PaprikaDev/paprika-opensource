from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from ragu.utils.tools import tools

# Run a tool ordered by the model
def call_tool(state):
    # Define a tool node
    tool_node = ToolNode(tools)
    # Define the tool calls
    tool_calls = state['messages'][-1].tool_calls
    # Call the tools and return
    return tool_node.invoke(tool_calls)