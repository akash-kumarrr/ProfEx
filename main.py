import json
from typing import Any
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Import your tools 
from tools import ArxivSearch, WikiSearch

load_dotenv()

TOOL_MAP = {
    "ArxivSearch": ArxivSearch,
    "WikiSearch": WikiSearch
}

class State(TypedDict):
    argument: str
    result: Any 

llm = ChatGroq(model="llama-3.3-70b-versatile").bind(
    response_format={"type": "json_object"}
)

def SingleFunctionNode(state: State) -> State:
    print(f"--- Processing Query: '{state['argument']}' ---")
    
    system_prompt = (
        "You are an routing assistant. Your job is to select the correct tool based on user input.\n"
        "Available Tools:\n"
        "- Use 'ArxivSearch' if the user is asking for research papers or academic studies.\n"
        "- Use 'WikiSearch' if the user is asking for general knowledge, history, or definitions.\n\n"
        "You must respond ONLY with a JSON object in this exact format:\n"
        '{"tool": "ToolName", "query": "extracted search query text"}'
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state['argument'])
    ]
    
    response = llm.invoke(messages)
    
    decision = json.loads(response.content)
    chosen_tool = decision.get("tool")
    search_query = decision.get("query")
    
    print(f"--- LLM Selected: {chosen_tool} with Query: '{search_query}' ---")
    
    if chosen_tool in TOOL_MAP:
        tool_output = TOOL_MAP[chosen_tool].invoke(search_query)
        return {"argument": state['argument'], "result": tool_output}
    
    return {"argument": state['argument'], "result": "Error: No valid tool was chosen by the LLM."}

workflow = StateGraph(State)

workflow.add_node("function_node", SingleFunctionNode)

workflow.add_edge(START, "function_node")
workflow.add_edge("function_node", END)

app = workflow.compile()

output = app.invoke({"argument": "What is the latest breakthrough in LLM reasoning on Arxiv?"})

print("\n--- FINAL GRAPH OUTPUT ---")
print(output["result"])