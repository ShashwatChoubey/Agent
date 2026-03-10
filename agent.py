import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langgraph.graph import StateGraph,END,START
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

class AgentState(TypedDict):
    question:str
    result:str
    

def node1(state: AgentState):
    response = tavily_client.search(state["question"])
    return {"result":response}

def node2(state:AgentState):
    messages = [(
        "human",
        f"Summarize this research:{state['result']}"
    )]
    output = model.invoke(messages)
    return {"result":output.content}

builder = StateGraph(AgentState)
builder.add_node("node_1",node1)
builder.add_node("node_2",node2)
builder.add_edge(START,"node_1")
builder.add_edge("node_1","node_2")
builder.add_edge("node_2",END)

graph = builder.compile()
if __name__ == "__main__":
    test = graph.invoke({"question":"What is the current crude oil price now"})
    print(test["result"])
