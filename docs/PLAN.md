# ml-finance-agent - Development Plan

## Objective

AI agent for credit risk analysis. Users interact in natural language and the agent queries data, makes predictions, and explains decisions.

## Demo

```
User: "Analyze client 1234. Is this a high risk?"

Agent: "Client 1234 has HIGH default risk (78% probability).
Key factors:
- Monthly income: $2,500 (below average of $4,200)
- 3 active loans
- Payment delay history: 45 days
Recommendation: deny credit or require collateral."
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Frontend (HTML)                    │
│              Chat interface + Dashboard               │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  FastAPI + Guardrails                 │
│           Input validation + PII detection            │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│              ReAct Agent (LangChain)                  │
│                   3+ Tools:                           │
│  ┌─────────────┐ ┌──────────┐ ┌───────────────────┐ │
│  │ predict_risk │ │ query_db │ │ explain_decision  │ │
│  │ (ML model)  │ │  (RAG)   │ │  (LLM reasoning)  │ │
│  └─────────────┘ └──────────┘ └───────────────────┘ │
└──────────────────────┬──────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌──────────────┐ ┌──────────┐ ┌──────────────┐
│  ML Model    │ │ Vector   │ │ LLM API      │
│  (sklearn)   │ │ Store    │ │ (Groq/OpenAI)│
│  + MLflow    │ │ (FAISS)  │ │              │
└──────────────┘ └──────────┘ └──────────────┘
```

## Repository Structure

```
ml-finance-agent/
├── .github/workflows/ci.yml       # CI/CD
├── data/
│   ├── raw/                       # Original dataset (DVC)
│   ├── processed/                 # Processed data
│   └── golden_set/                # 20+ query/answer pairs for evaluation
├── src/
│   ├── features/
│   │   └── feature_engineering.py # Feature transformations
│   ├── models/
│   │   ├── baseline.py            # Scikit-learn baseline
│   │   ├── mlp.py                 # MLP PyTorch
│   │   └── train.py               # Training pipeline with MLflow
│   ├── agent/
│   │   ├── react_agent.py         # ReAct agent
│   │   ├── tools.py               # 3+ custom tools
│   │   └── rag_pipeline.py        # RAG: embedding + retriever
│   ├── serving/
│   │   ├── app.py                 # FastAPI
│   │   └── Dockerfile             # Container
│   ├── monitoring/
│   │   ├── drift.py               # Evidently drift detection
│   │   └── metrics.py             # Custom metrics
│   └── security/
│       ├── guardrails.py          # Input/output guardrails
│       └── pii_detection.py       # Sensitive data detection
├── tests/
│   ├── conftest.py                # Shared fixtures
│   ├── test_features.py           # Feature engineering tests
│   ├── test_models.py             # Model tests
│   ├── test_agent.py              # Agent tests
│   ├── test_api.py                # API tests
│   └── test_guardrails.py         # Security tests
├── evaluation/
│   ├── ragas_eval.py              # RAGAS: 4 metrics
│   ├── llm_judge.py               # LLM-as-judge: 3+ criteria
│   └── ab_test_prompts.py         # A/B test prompts
├── notebooks/
│   └── 01_eda.ipynb               # Exploratory data analysis
├── docs/
│   ├── PLAN.md                    # This file
│   ├── MODEL_CARD.md              # Model Card
│   ├── SYSTEM_CARD.md             # System Card
│   ├── LGPD_PLAN.md               # LGPD compliance plan
│   ├── OWASP_MAPPING.md           # 5+ mapped threats
│   └── RED_TEAM_REPORT.md         # 5+ adversarial scenarios
├── configs/
│   ├── model_config.yaml          # Hyperparameters
│   └── monitoring_config.yaml     # Drift thresholds
├── docker-compose.yml             # Local orchestration
├── dvc.yaml                       # DVC pipeline
├── pyproject.toml                 # Dependencies
├── Makefile                       # Shortcuts: make train, make serve, make test
└── README.md
```

## Development Stages

### Stage 1 — Data + Baseline (Phases 01-02) — ~3 sessions

| Step | Task | Deliverable |
|------|------|-------------|
| 1.1 | Download credit risk dataset from Kaggle | `data/raw/` |
| 1.2 | EDA notebook (distributions, correlations, missing values) | `notebooks/01_eda.ipynb` |
| 1.3 | Feature engineering (derived features, encoding) | `src/features/feature_engineering.py` |
| 1.4 | Scikit-learn baseline (LogisticRegression + RandomForest) | `src/models/baseline.py` |
| 1.5 | MLP with PyTorch (simple neural network) | `src/models/mlp.py` |
| 1.6 | Standardized MLflow tracking (params, metrics, tags, artifacts) | `src/models/train.py` |
| 1.7 | DVC configured | `dvc.yaml` |
| 1.8 | Feature and model tests | `tests/` |

### Stage 2 — LLM + Agent (Phases 03+05) — ~4 sessions

| Step | Task | Deliverable |
|------|------|-------------|
| 2.1 | Configure LLM via API (Groq or OpenAI) | Config |
| 2.2 | Tool 1: `predict_risk` (calls ML model) | `src/agent/tools.py` |
| 2.3 | Tool 2: `query_database` (SQL/pandas query on data) | `src/agent/tools.py` |
| 2.4 | Tool 3: `explain_decision` (explains via SHAP or LLM) | `src/agent/tools.py` |
| 2.5 | RAG pipeline (data embeddings + FAISS vector store) | `src/agent/rag_pipeline.py` |
| 2.6 | ReAct agent with LangChain | `src/agent/react_agent.py` |
| 2.7 | FastAPI `/chat` endpoint | `src/serving/app.py` |
| 2.8 | Chat frontend interface | `static/index.html` |
| 2.9 | CI/CD with GitHub Actions | `.github/workflows/ci.yml` |
| 2.10 | Docker + docker-compose | `Dockerfile`, `docker-compose.yml` |

### Stage 3 — Evaluation + Observability (Phases 03-05) — ~2 sessions

| Step | Task | Deliverable |
|------|------|-------------|
| 3.1 | Create golden set (20+ query/expected answer pairs) | `data/golden_set/` |
| 3.2 | RAGAS evaluation (faithfulness, relevancy, precision, recall) | `evaluation/ragas_eval.py` |
| 3.3 | LLM-as-judge (3+ evaluation criteria) | `evaluation/llm_judge.py` |
| 3.4 | Drift detection with Evidently (PSI thresholds) | `src/monitoring/drift.py` |
| 3.5 | Metrics endpoint (`/metrics`) | `src/monitoring/metrics.py` |
| 3.6 | Agent and evaluation tests | `tests/` |

### Stage 4 — Security + Governance (Phases 04-05) — ~2 sessions

| Step | Task | Deliverable |
|------|------|-------------|
| 4.1 | Input guardrails (prompt injection, max length) | `src/security/guardrails.py` |
| 4.2 | Output guardrails (PII detection, sanitization) | `src/security/pii_detection.py` |
| 4.3 | OWASP Top 10 mapping (5+ threats) | `docs/OWASP_MAPPING.md` |
| 4.4 | Red teaming (5+ adversarial scenarios) | `docs/RED_TEAM_REPORT.md` |
| 4.5 | LGPD compliance plan | `docs/LGPD_PLAN.md` |
| 4.6 | Model Card + System Card | `docs/MODEL_CARD.md`, `docs/SYSTEM_CARD.md` |
| 4.7 | Security tests | `tests/test_guardrails.py` |

### Stage 5 — Polish + Deploy — ~1 session

| Step | Task | Deliverable |
|------|------|-------------|
| 5.1 | Final README with screenshots and demo link | `README.md` |
| 5.2 | Deploy to Railway | Public URL |
| 5.3 | Makefile (make train, make serve, make test) | `Makefile` |

## Tech Stack

| Layer | Tool |
|-------|------|
| Classic ML | scikit-learn, PyTorch |
| LLM | Groq (Llama) or OpenAI (GPT-4o-mini) |
| Agent | LangChain (ReAct) |
| RAG | FAISS + sentence-transformers |
| API | FastAPI |
| Tracking | MLflow |
| Data | DVC |
| Tests | pytest (coverage >= 60%) |
| Drift | Evidently |
| Security | Presidio (PII), custom guardrails |
| Evaluation | RAGAS, LLM-as-judge |
| Container | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Deploy | Railway |

## Estimate

| Stage | Sessions (~2h each) |
|-------|---------------------|
| 1 - Data + Baseline | 3 |
| 2 - LLM + Agent | 4 |
| 3 - Evaluation | 2 |
| 4 - Security | 2 |
| 5 - Polish + Deploy | 1 |
| **Total** | **~12 sessions** |

## Datathon Coverage

| Criteria (weight) | Covered? |
|-------------------|----------|
| Pipeline + baseline (10%) | Yes |
| LLM serving + agent (15%) | Yes |
| Quality evaluation (10%) | Yes |
| Observability + monitoring (10%) | Yes |
| Security + guardrails (10%) | Yes |
| Governance + compliance (5%) | Yes |
| Documentation + architecture (5%) | Yes |
| Business criteria (30%) | Depends on company brief |

**70% of technical points fully covered before receiving the datathon brief.**
