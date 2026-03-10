from fastapi import FastAPI
from pydantic import BaseModel
from agent import graph

app = FastAPI()

class Input(BaseModel):
    question:str

@app.get('/')
def health():
    return {"status":"ok"}

@app.post('/agent')
def agent(input:Input):
    result = graph.invoke({"question":input.question})
    return {"result":result["result"][0]["text"]}
