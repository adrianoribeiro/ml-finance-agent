import time

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.agent.react_agent import chat
from src.monitoring.metrics import record_latency, get_metrics
from src.security.guardrails import InputGuardrail, OutputGuardrail
from src.serving.dashboard import DASHBOARD_HTML

_DESCRIPTION = """
## Credit Risk AI Agent API

API de análise de risco de crédito com IA explicável baseada em modelos de ML e RAG.

### Funcionalidades
- **Predição de risco** — probabilidade de inadimplência calculada pelo modelo treinado
- **Consulta de dados** — estatísticas agregadas sobre a base de clientes
- **Explicabilidade** — importância das features para cada decisão (coeficientes do modelo)
- **Busca documental** — RAG sobre o dicionário de dados e documentação do domínio

### Segurança
Todas as mensagens passam por **guardrails** de entrada (injeção de prompt, tópico fora de escopo,
limite de 4096 caracteres) e **sanitização** de saída (redação de CPF, telefone e e-mail — LGPD).
"""

_TAGS = [
    {
        "name": "Chat",
        "description": "Interação conversacional com o agente ReAct de risco de crédito.",
    },
    {
        "name": "Observabilidade",
        "description": "Endpoints de saúde, métricas de latência/predição e dashboard de monitoramento.",
    },
]

app = FastAPI(
    title="Credit Risk AI Agent",
    description=_DESCRIPTION,
    version="1.0.0",
    openapi_tags=_TAGS,
    contact={
        "name": "ML Finance Agent",
        "url": "https://github.com/fabricio/ml-finance-agent",
    },
    license_info={
        "name": "MIT",
    },
)

input_guard = InputGuardrail()
output_guard = OutputGuardrail()


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Pergunta ou instrução em linguagem natural para o agente de risco de crédito.",
        examples=[
            "Qual a probabilidade de inadimplência de um cliente com renda de 5000, "
            "2 dependentes e dívida rotativa de 30%?"
        ],
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": (
                        "Qual a probabilidade de inadimplência de um cliente com renda de 5000, "
                        "2 dependentes e dívida rotativa de 30%?"
                    )
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    response: str = Field(
        ...,
        description=(
            "Resposta gerada pelo agente. Pode conter análise de risco, estatísticas, "
            "explicações de decisão ou resultado de busca documental. "
            "Dados pessoais (CPF, telefone, e-mail) são redatados automaticamente (LGPD)."
        ),
        examples=[
            "A probabilidade de inadimplência estimada é de **23,4%** (risco moderado). "
            "Os principais fatores são: utilização de crédito rotativo (42%), "
            "número de linhas de crédito em aberto (8) e tempo de emprego (3 anos)."
        ],
    )


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
    summary="Enviar mensagem ao agente",
    response_description="Resposta gerada pelo agente ReAct após execução das ferramentas necessárias.",
    responses={
        200: {
            "description": "Resposta do agente (inclui casos onde o guardrail rejeitou a entrada).",
            "content": {
                "application/json": {
                    "examples": {
                        "risco_calculado": {
                            "summary": "Predição de risco",
                            "value": {
                                "response": (
                                    "A probabilidade de inadimplência estimada é de **23,4%** (risco moderado). "
                                    "Os principais fatores são: utilização de crédito rotativo (42%)."
                                )
                            },
                        },
                        "entrada_rejeitada": {
                            "summary": "Guardrail bloqueou a entrada",
                            "value": {"response": "Suspicious input detected."},
                        },
                        "fora_de_escopo": {
                            "summary": "Tópico fora do escopo",
                            "value": {
                                "response": "This topic is outside the scope of credit risk analysis."
                            },
                        },
                    }
                }
            },
        },
        422: {"description": "Corpo da requisição inválido (validação Pydantic)."},
    },
)
def chat_endpoint(request: ChatRequest):
    """
    Envia uma mensagem ao agente ReAct de risco de crédito e retorna a resposta gerada.

    O agente utiliza as seguintes ferramentas conforme necessário:
    - **predict_risk** — calcula a probabilidade de inadimplência via modelo ML
    - **query_data** — consulta estatísticas da base de clientes
    - **explain_decision** — explica a decisão por importância de features
    - **search_docs** — busca semântica na documentação (RAG + FAISS)

    **Guardrails aplicados:**
    - Entrada: bloqueio de injeção de prompt, tópicos fora de escopo e mensagens > 4096 chars
    - Saída: redação automática de CPF, telefone e e-mail (LGPD)
    """
    is_valid, reason = input_guard.validate(request.message)
    if not is_valid:
        return ChatResponse(response=reason)

    start = time.time()
    result = chat(request.message)
    duration = (time.time() - start) * 1000
    record_latency("/chat", duration)

    result = output_guard.sanitize(result)
    return ChatResponse(response=result)


@app.get(
    "/health",
    tags=["Observabilidade"],
    summary="Verificação de saúde",
    response_description="Status atual do serviço.",
    responses={
        200: {
            "description": "Serviço operacional.",
            "content": {"application/json": {"example": {"status": "ok"}}},
        }
    },
)
def health():
    """Retorna `{"status": "ok"}` quando o serviço está em execução e pronto para receber requisições."""
    return {"status": "ok"}


@app.get(
    "/metrics",
    tags=["Observabilidade"],
    summary="Métricas de monitoramento",
    response_description="Resumo de latência e predições registradas na sessão atual.",
    responses={
        200: {
            "description": "Métricas agregadas da sessão.",
            "content": {
                "application/json": {
                    "example": {
                        "uptime_seconds": 3600,
                        "total_predictions": 42,
                        "avg_probability": 0.18,
                        "risk_distribution": {"low": 30, "medium": 10, "high": 2},
                        "latency": {
                            "/chat": {
                                "count": 42,
                                "avg_ms": 1240.5,
                                "min_ms": 320.1,
                                "max_ms": 4800.0,
                            }
                        },
                    }
                }
            },
        }
    },
)
def metrics():
    """
    Retorna métricas em memória coletadas desde o início da sessão:

    - **uptime_seconds** — tempo de atividade do serviço
    - **total_predictions** — número de predições de risco realizadas
    - **avg_probability** — probabilidade média de inadimplência
    - **risk_distribution** — contagem por nível de risco (low / medium / high)
    - **latency** — estatísticas de latência por endpoint (count, avg, min, max em ms)
    """
    return get_metrics()


@app.get(
    "/dashboard",
    response_class=HTMLResponse,
    tags=["Observabilidade"],
    summary="Dashboard de monitoramento",
    response_description="Página HTML interativa com gráficos de métricas em tempo real.",
    responses={
        200: {"description": "Página HTML do dashboard de monitoramento."}
    },
)
def dashboard():
    """
    Serve o dashboard HTML de monitoramento com gráficos de latência,
    distribuição de risco e volume de predições atualizados em tempo real.
    """
    return DASHBOARD_HTML
