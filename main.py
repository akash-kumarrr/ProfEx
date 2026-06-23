import json
from typing import Any
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from tools import ArxivSearch, WikiSearch

load_dotenv()

# Note: Added WikiSearch back here in case the LLM decides to fallback to it
TOOL_MAP = {
    "ArxivSearch": ArxivSearch
}

# Initializes the local Transformer pipeline

class State(TypedDict):
    argument: str
    result: Any 

llm = ChatGroq(model="llama-3.3-70b-versatile").bind(
    response_format={"type": "json_object"}
)

def SingleFunctionNode(state: State) -> State:
    print(f"--- Processing Query: '{state['argument']}' ---")
    system_prompt = (
        "You are a routing assistant. Your job is to select the correct tool based on user input.\n"
        "Available Tools:\n"
        "- Use 'ArxivSearch' if the user ask something and should return the summarized text of that output... in length between 100 to 500 texts or length .\n"
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

# Interactive prompt check
query = str(input("Query : "))
output = app.invoke({"argument": query})

print("\n--- FINAL GRAPH OUTPUT (TRANSFORMER SUMMARY) ---")
print(output["result"])