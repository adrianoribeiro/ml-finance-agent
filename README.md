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
│   │   └── app.py                   # FastAPI endpoints (/chat, /health, /metrics)
│   ├── monitoring/
│   │   ├── drift.py                 # PSI-based drift detection
│   │   └── metrics.py               # Operational metrics (latency, predictions)
│   └── security/
│       └── guardrails.py            # Input validation + PII redaction
├── tests/                           # pytest (27 tests, 83% coverage)
├── evaluation/
│   ├── ragas_eval.py                # RAGAS: 4 metrics
│   └── llm_judge.py                 # LLM-as-judge: 3 criteria
├── notebooks/
│   ├── 01_eda.ipynb                 # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb # Feature engineering pipeline
│   ├── 03_baseline_model.ipynb      # LogReg + RandomForest baselines
│   └── 04_mlp_pytorch.ipynb         # MLP neural network
├── data/
│   ├── raw/                         # Original dataset
│   ├── processed/                   # Clean dataset (149,986 x 12)
│   ├── docs/                        # Documents for RAG
│   └── golden_set/                  # 22 evaluation pairs
├── docs/
│   ├── PLAN.md                      # Development roadmap
│   ├── OWASP_MAPPING.md             # 6 threats mapped
│   └── RED_TEAM_REPORT.md           # 6 adversarial scenarios
└── .github/workflows/ci.yml        # CI/CD: lint + tests
```

## Model Results

| Model | AUC |
|-------|-----|
| MLP PyTorch | 0.8675 |
| Logistic Regression | 0.8620 |
| Random Forest | 0.8387 |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run notebooks in order (01, 02, 03, 04):

```bash
jupyter notebook notebooks/
```

View MLflow experiments:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Agent

Create a `.env` file with your OpenRouter key:

```
OPENROUTER_API_KEY=sk-or-...
```

Test in terminal:

```bash
python3 -c "from src.agent.react_agent import chat; print(chat('Qual o risco de um cliente de 30 anos com renda 2000?'))"
```

Start the API:

```bash
uvicorn src.serving.app:app --reload
```

Test the API:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Qual o risco de um cliente de 30 anos com renda 2000?"}'
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
