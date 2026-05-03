# Credit Risk AI Agent

AI-powered agent for credit risk analysis. Predicts whether a client will default on their financial obligations within the next 2 years, and explains the reasoning in natural language.

## How the Dataset Was Built

```
2024                                            2026
  │                                               │
  │  Collected data from 150k clients:            │  Checked:
  │  - income                                     │  - Client 1: paid everything ✓ → 0
  │  - age                                        │  - Client 2: defaulted ✗ → 1
  │  - debts                                      │  - Client 3: paid everything ✓ → 0
  │  - payment delays                             │  - Client 4: defaulted ✗ → 1
  │  - etc                                        │
```

Now a new client arrives. We have their data today but don't know the future. The model uses patterns learned from 150k historical clients to predict: will this new client default or not?

This is supervised learning with a binary target:
- **0** = client will pay (no default)
- **1** = client will default

## Project Structure

```
ml-finance-agent/
├── src/
│   ├── features/
│   │   └── feature_engineering.py   # Data cleaning and feature creation
│   ├── models/
│   │   ├── baseline.py              # LogReg, RandomForest, MLP PyTorch
│   │   └── train.py                 # Training pipeline with MLflow
│   ├── agent/
│   │   ├── react_agent.py           # ReAct agent (LangChain + GPT-4o-mini)
│   │   ├── tools.py                 # predict_risk, query_data, explain_decision, search_docs
│   │   └── rag_pipeline.py          # FAISS vector store + sentence-transformers
│   ├── serving/
│   │   └── app.py                   # FastAPI endpoints (/chat, /health, /metrics, /dashboard)
│   ├── monitoring/
│   │   ├── drift.py                 # PSI-based drift detection
│   │   └── metrics.py               # Operational metrics (latency, predictions)
│   └── security/
│       └── guardrails.py            # Input validation + PII redaction
├── tests/                           # pytest (27 tests, 83% coverage)
├── evaluation/
│   ├── ragas_eval.py                # RAGAS: 4 metrics
│   ├── llm_judge.py                 # LLM-as-judge: 3 criteria
│   └── benchmark.py                 # Benchmark: 3 agent configurations
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb # Feature engineering pipeline
│   ├── 03_baseline_model.ipynb      # LogReg + RandomForest baselines
│   └── 04_mlp_pytorch.ipynb         # MLP neural network
├── data/
│   ├── raw/                         # Original dataset (DVC)
│   ├── processed/                   # Clean dataset (DVC)
│   ├── docs/                        # Documents for RAG
│   └── golden_set/                  # 22 evaluation pairs
├── models/                          # Trained model + scaler + FAISS index (DVC)
├── docs/
│   ├── PLAN.md                      # Development roadmap
│   ├── MODEL_CARD.md                # Model documentation
│   ├── SYSTEM_CARD.md               # System documentation
│   ├── LGPD_PLAN.md                 # LGPD compliance plan
│   ├── OWASP_MAPPING.md             # 6 threats mapped
│   └── RED_TEAM_REPORT.md           # 6 adversarial scenarios
├── Dockerfile                       # Container image
├── docker-compose.yml               # Local orchestration
├── dvc.yaml                         # Data versioning pipeline
└── .github/workflows/ci.yml        # CI/CD: lint + tests
```

## Model Results

| Model | AUC |
|-------|-----|
| MLP PyTorch | 0.8675 |
| Logistic Regression | 0.8620 |
| Random Forest | 0.8387 |

## Setup

### 1. Clone and install

```bash
git clone git@github.com:adrianoribeiro/ml-finance-agent.git
cd ml-finance-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Download data and models (DVC + S3)

Configure AWS credentials (ask the team for the access key):

```bash
aws configure
# AWS Access Key ID: <provided by team>
# AWS Secret Access Key: <provided by team>
# Default region: sa-east-1
# Default output: json
```

Pull data and model files:

```bash
dvc pull
```

This downloads `data/raw/`, `data/processed/` and `models/` (trained model, scaler, FAISS index).

### 3. Configure API key

Create a `.env` file with your OpenRouter key:

```
OPENROUTER_API_KEY=sk-or-...
```

### 4. Run the agent

```bash
uvicorn src.serving.app:app --reload
```

Test:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Qual o risco de um cliente de 30 anos com renda 2000?"}'
```

Dashboard: http://localhost:8000/dashboard

## Notebooks (optional)

The notebooks document the exploration and training process. Run in order (01, 02, 03, 04):

```bash
jupyter notebook notebooks/
```

After running notebooks 03 and 04, view experiments in MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Note: `mlflow.db` is generated locally when you run the notebooks. It is not included in the repository.

## Docker

```bash
docker compose up --build
```

## Tests

```bash
pytest tests/ -v --cov
```

## Dataset Features

| Feature | Description |
|---------|-------------|
| **SeriousDlqin2yrs** | TARGET — client defaulted within 2 years (0=no, 1=yes) |
| RevolvingUtilizationOfUnsecuredLines | Credit card + personal credit usage ratio (balance / limit) |
| age | Client age |
| DebtRatio | Monthly debt payments / monthly income |
| MonthlyIncome | Monthly income |
| NumberOfOpenCreditLinesAndLoans | Number of open credit lines and loans |
| NumberRealEstateLoansOrLines | Number of real estate loans |
| NumberOfDependents | Number of dependents |
| income_missing | Flag: monthly income was missing in original data |
| total_late_payments | Sum of all late payment counts (30-59 + 60-89 + 90+ days) |
| has_late_payment | Binary: any late payment history |
| has_severe_late | Binary: 90+ days late at least once |

Source: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — Kaggle competition dataset with 150,000 samples. Default rate: 6.68%.
