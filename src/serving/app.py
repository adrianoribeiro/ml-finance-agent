import time

from fastapi import FastAPI
from pydantic import BaseModel

from src.agent.react_agent import chat
from src.monitoring.metrics import record_latency, get_metrics

app = FastAPI(title="Credit Risk Agent")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    start = time.time()
    result = chat(request.message)
    duration = (time.time() - start) * 1000
    record_latency("/chat", duration)
    return ChatResponse(response=result)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return get_metrics()
