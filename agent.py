import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langgraph.graph import StateGraph,END,START
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict,List

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


class AgentState(TypedDict):
    question:str
    result:str
    history:List[str]
    

def node1(state: AgentState):
    response = tavily_client.search(state["question"])
    return {"result":response}

def node2(state:AgentState):
    his = "\n".join(state["history"])
    

    messages = [(
        "human",
        f"Previous contect:{his} \n Summarize this research:{state['result']}+"
    )]
    output = model.invoke(messages)
    return {"result":output.content,"history":state["history"]+[f"Q:{state['question']} A:{output.content}"]}

builder = StateGraph(AgentState)
builder.add_node("node_1",node1)
builder.add_node("node_2",node2)
builder.add_edge(START,"node_1")
builder.add_edge("node_1","node_2")
builder.add_edge("node_2",END)

graph = builder.compile()
if __name__ == "__main__":
    test = graph.invoke({"question":"What is the current crude oil price now", "history":[]})
    test2 = graph.invoke({"question":"Why did it change recently", "history":test["history"]})
    print(test2["history"])
