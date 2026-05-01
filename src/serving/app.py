import time

from fastapi import FastAPI
from pydantic import BaseModel

from src.agent.react_agent import chat
from src.monitoring.metrics import record_latency, get_metrics
from fastapi.responses import HTMLResponse
from src.security.guardrails import InputGuardrail, OutputGuardrail
from src.serving.dashboard import DASHBOARD_HTML

app = FastAPI(title="Credit Risk Agent")

input_guard = InputGuardrail()
output_guard = OutputGuardrail()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    # Validate input
    is_valid, reason = input_guard.validate(request.message)
    if not is_valid:
        return ChatResponse(response=reason)

    start = time.time()
    result = chat(request.message)
    duration = (time.time() - start) * 1000
    record_latency("/chat", duration)

    # Sanitize output
    result = output_guard.sanitize(result)
    return ChatResponse(response=result)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return get_metrics()


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASHBOARD_HTML
