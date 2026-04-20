from fastapi import FastAPI
from pydantic import BaseModel

from src.agent.react_agent import chat

app = FastAPI(title="Credit Risk Agent")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    result = chat(request.message)
    return ChatResponse(response=result)


@app.get("/health")
def health():
    return {"status": "ok"}
