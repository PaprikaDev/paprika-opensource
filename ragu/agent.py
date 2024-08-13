graph = StateGraph(AgentState)

graph.add_node("scrape_llm", self.openai_inference_scrape)
graph.add_node("generate_llm", self.openai_inference_generate)
graph.add_node("action", self.call_tool)

# If the record already exists move to Augmentation + Generation
# Else scrape the record with the Retrieval agent
graph.add_conditional_edges(
    "scrape_llm",
    self.exists_record,
    {True: "generate_llm", False: "scrape_llm"}
)
graph.add_conditional_edges(
    "scrape_llm",
    self.requires_tool,
    {True: "action", False: "generate_llm"}
)

graph.add_edge("action", "scrape_llm")
graph.add_edge("generate_llm", END)
graph.set_entry_point("scrape_llm")

self.graph = graph.compile()
self.tools = {t.name: t for t in tools}
self.model = model.bind_tools(tools)