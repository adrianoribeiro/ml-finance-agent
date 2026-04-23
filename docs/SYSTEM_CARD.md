# System Card — Credit Risk AI Agent

## System Overview

Interactive AI agent for credit risk analysis. Users ask questions in natural language and the agent uses ML models, data analysis, and document retrieval to respond.

## Architecture

```
User → FastAPI (/chat) → Input Guardrail → ReAct Agent → Tools → Response → Output Guardrail → User
                                               │
                                    ┌──────────┼──────────────┐
                                    ▼          ▼              ▼
                              predict_risk  query_data   search_docs
                              (ML model)    (pandas)     (FAISS/RAG)
```

## Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Agent | LangChain + GPT-4o-mini (OpenRouter) | Reasoning and tool selection |
| ML Model | Scikit-learn Logistic Regression | Default probability prediction |
| RAG | FAISS + sentence-transformers (all-MiniLM-L6-v2) | Document retrieval for domain questions |
| API | FastAPI | HTTP endpoint for chat interaction |
| Monitoring | PSI drift detection + custom metrics | Data drift and operational monitoring |
| Security | InputGuardrail + OutputGuardrail | Prompt injection prevention + PII redaction |

## Tools

| Tool | Input | Output |
|------|-------|--------|
| predict_risk | Client features (JSON) | Default probability + risk level |
| query_data | Natural language question | Dataset statistics |
| explain_decision | Client features (JSON) | Top 5 contributing factors |
| search_docs | Natural language query | Relevant document excerpts |

## Security Measures

- **Input validation**: Prompt injection detection, off-topic filtering, max length (4096 chars)
- **Output sanitization**: PII redaction (CPF, email, phone)
- **Read-only tools**: No tool can modify data or external systems
- **API key management**: OpenRouter key stored in `.env`, never committed to git

See `docs/OWASP_MAPPING.md` for detailed threat mapping and `docs/RED_TEAM_REPORT.md` for adversarial testing results.

## Evaluation

- **Golden set**: 22 question/answer pairs covering all tools
- **RAGAS**: 4 automated metrics (faithfulness, relevancy, context precision, context recall)
- **LLM-as-judge**: 3 criteria (accuracy, clarity, completeness) scored 1-10

## Limitations

- Depends on external LLM API (OpenRouter) — latency and availability not controlled
- RAG knowledge limited to indexed documents (data dictionary)
- Agent may hallucinate if question falls outside tool capabilities
- No real-time data — predictions based on static training dataset

## Human Oversight

This system is a decision-support tool. Credit decisions must involve human review. The agent provides probability estimates and explanations to inform — not replace — the analyst's judgment.
