import os
from dotenv import load_dotenv
from tavily import TavilyClient
from langgraph.graph import StateGraph,END,START
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict,List

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


class AgentState(TypedDict):
    question:str
    result:str
    history:List[str]
    route:str

def router(state:AgentState):
    messages = [(
        "human",
        f"For you to give ans this {state['question']} do you need web search.If you do return only true or false"
    )]
    response = model.invoke(messages)
    response= response.content
    if ("true" in response):
        return {"route":"search"}
    return {"route":"direct"}


def node1(state: AgentState):
    response = tavily_client.search(state["question"])
    return {"result":response}

def node2(state:AgentState):
    his = "\n".join(state["history"])
    result = state.get('result', '')
    
    if result:
        messages = [(
            "human",
            f"Previous context:{his} \n Summarize this research:{result}"
        )]
    else:
        messages = [(
            "human",
            f"Answer this question:{state['question']}"
        )]
    
    output = model.invoke(messages)
    return {"result":output.content,"history":state["history"]+[f"Q:{state['question']} A:{output.content}"]}







builder = StateGraph(AgentState)
builder.add_node("node_1",node1)
builder.add_node("node_2",node2)
builder.add_node("router",router)
builder.add_edge(START,"router")
builder.add_conditional_edges("router",lambda state: state["route"],{"direct":"node_2","search":"node_1"})

builder.add_edge("node_1","node_2")
builder.add_edge("node_2",END)

graph = builder.compile()
if __name__ == "__main__":
    test = graph.invoke({"question":"What is 2+2", "history":[], "route":""})
    test2 = graph.invoke({"question":"What is the current crude oil price", "history":[], "route":""})
    print("Q1 route:", test["route"])
    print("Q2 route:", test2["route"])