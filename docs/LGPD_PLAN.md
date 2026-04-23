# LGPD Compliance Plan

Lei Geral de Proteção de Dados (Lei 13.709/2018) — compliance plan for the Credit Risk AI Agent.

## Data Classification

| Data | Classification | Treatment |
|------|---------------|-----------|
| Training dataset (cs-training.csv) | Anonymized | No personal identifiers. Clients identified only by index. |
| Model predictions | Non-personal | Probability scores, no PII. |
| Agent conversations | Potentially personal | User may include personal data in queries. |

## LGPD Principles Applied

### 1. Purpose (Finalidade)

Data is used exclusively for credit risk analysis. The model predicts default probability to support credit decisions.

### 2. Adequacy (Adequação)

Only relevant features are used (income, age, payment history). No collection of sensitive data (ethnicity, religion, political opinion).

### 3. Necessity (Necessidade)

Feature engineering reduced the dataset from 11 to 11 features (3 original late payment columns merged into 3 engineered features). Only necessary information is retained.

### 4. Free Access (Livre Acesso)

The `explain_decision` tool allows users to understand which factors influenced a prediction. Model Card documents the model's behavior.

### 5. Data Quality (Qualidade dos Dados)

- 14 records with invalid ages removed
- Missing values imputed with median (documented in feature engineering)
- Outliers capped at defined thresholds
- Drift detection monitors data quality over time

### 6. Security (Segurança)

- API key stored in `.env` (not in repository)
- Input guardrails prevent prompt injection
- Output guardrails redact PII (CPF, email, phone)
- No raw client data exposed through API

### 7. Non-Discrimination (Não Discriminação)

- `class_weight='balanced'` prevents the model from ignoring the minority class
- No protected attributes (gender, race) used as features
- Age and income are used as risk factors, which is standard practice in credit analysis and permitted by regulation

## Data Subject Rights

| Right | Implementation |
|-------|---------------|
| Right to explanation | `explain_decision` tool provides feature-level explanations |
| Right to access | Model Card and System Card publicly document system behavior |
| Right to deletion | Training data is anonymized; no individual records can be traced |
| Right to correction | Drift detection flags when data distributions change |

## Data Retention

- Training data: retained for model reproducibility and audit
- Model artifacts: versioned in MLflow with metadata
- Agent conversations: not persisted (in-memory only, lost on restart)

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|-----------|
| PII leakage in agent output | Low | High | OutputGuardrail redacts CPF, email, phone |
| Model bias | Medium | High | Balanced class weights, no protected attributes |
| Data breach | Low | High | No PII in dataset, API key in .env |
| Unauthorized access | Low | Medium | API can be protected with authentication middleware |
