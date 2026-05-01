.DEFAULT_GOAL := help
PYTHON        := python3
VENV          := .venv
VENV_PYTHON   := $(VENV)/bin/python
VENV_PIP      := $(VENV)/bin/pip

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help
help: ## Show this help message
	@echo ""
	@echo "\033[1mCredit Risk AI Agent\033[0m — Makefile commands"
	@echo ""
	@echo "\033[1;33m▶ SETUP\033[0m"
	@echo "  \033[36mvenv\033[0m          Create a Python virtual environment in .venv/"
	@echo "                  Safe to run multiple times — skips if .venv/ already exists."
	@echo ""
	@echo "  \033[36minstall\033[0m       Create .venv/ (if needed) and install all dependencies"
	@echo "                  including dev extras (pytest, ruff). Equivalent to:"
	@echo "                  make venv && .venv/bin/pip install -e '.[dev]'"
	@echo ""
	@echo "  \033[36msetup\033[0m         Alias for install — same behaviour."
	@echo ""
	@echo "  \033[36menv\033[0m           Create a .env file with a placeholder OPENROUTER_API_KEY."
	@echo "                  Required before running the server or evaluation."
	@echo "                  Skipped if .env already exists."
	@echo ""
	@echo "\033[1;33m▶ CODE QUALITY\033[0m"
	@echo "  \033[36mlint\033[0m          Lint src/ and tests/ with ruff (same check as CI)."
	@echo "                  Exits non-zero if any issues are found."
	@echo ""
	@echo "  \033[36mlint-fix\033[0m      Lint and auto-apply all safe fixes with ruff."
	@echo "                  Review the diff before committing."
	@echo ""
	@echo "\033[1;33m▶ TESTING\033[0m"
	@echo "  \033[36mtest\033[0m          Run the full test suite with XML coverage report."
	@echo "                  Mirrors CI exactly. Coverage target: 80% (currently ~83%)."
	@echo "                  Stops at the first failure (-x flag)."
	@echo ""
	@echo "  \033[36mtest-fast\033[0m     Run tests without coverage — faster inner loop."
	@echo "                  Use during active development; run 'make test' before committing."
	@echo ""
	@echo "  \033[36mtest-verbose\033[0m  Run tests with per-test output and terminal coverage summary."
	@echo "                  Shows which lines are not covered (--cov-report=term-missing)."
	@echo ""
	@echo "  \033[36mtest-features\033[0m Run tests/test_features.py — feature engineering transformations"
	@echo "                  (age capping, outlier handling, late payment features, etc.)."
	@echo ""
	@echo "  \033[36mtest-guardrails\033[0m Run tests/test_guardrails.py — input validation (injection"
	@echo "                  detection, length limits, off-topic filtering) and PII redaction"
	@echo "                  (CPF, phone, email — LGPD compliance)."
	@echo ""
	@echo "  \033[36mtest-models\033[0m   Run tests/test_models.py — model training and inference"
	@echo "                  for LogReg, RandomForest, and MLP baselines."
	@echo ""
	@echo "\033[1;33m▶ SERVING\033[0m"
	@echo "  \033[36mserve\033[0m         Start the FastAPI server with hot-reload on http://localhost:8000"
	@echo "                  Endpoints: POST /chat  GET /health  GET /metrics"
	@echo "                  Requires: models/ artifacts and .env with OPENROUTER_API_KEY."
	@echo ""
	@echo "  \033[36mserve-prod\033[0m    Start the server in production mode (0.0.0.0:8000, 4 workers)."
	@echo "                  No hot-reload. Use behind a reverse proxy (e.g. nginx)."
	@echo ""
	@echo "\033[1;33m▶ NOTEBOOKS & TRAINING\033[0m"
	@echo "  \033[36mnotebooks\033[0m     Open Jupyter in the notebooks/ directory."
	@echo "                  Run in order: 01_eda → 02_feature_engineering →"
	@echo "                  03_baseline_model → 04_mlp_pytorch."
	@echo "                  Generates models/ and data/processed/ artifacts needed by the server."
	@echo ""
	@echo "  \033[36mmlflow\033[0m        Open the MLflow UI at http://localhost:5000"
	@echo "                  Browse training experiments, compare runs, and view model metrics."
	@echo "                  Reads from mlflow.db (created by notebooks 03 and 04)."
	@echo ""
	@echo "\033[1;33m▶ EVALUATION\033[0m"
	@echo "  \033[36meval-ragas\033[0m    Run the RAGAS pipeline against the 22-pair golden set."
	@echo "                  Reports 4 metrics: faithfulness, answer relevancy,"
	@echo "                  context precision, context recall."
	@echo "                  Requires: .env with OPENROUTER_API_KEY and a running model server."
	@echo ""
	@echo "  \033[36meval-judge\033[0m    Run LLM-as-judge evaluation against the golden set."
	@echo "                  Scores each response on 3 criteria: accuracy, clarity, safety."
	@echo "                  Requires: .env with OPENROUTER_API_KEY."
	@echo ""
	@echo "  \033[36meval\033[0m          Run both eval-ragas and eval-judge sequentially."
	@echo ""
	@echo "\033[1;33m▶ CI\033[0m"
	@echo "  \033[36mci\033[0m            Run lint then test — full local CI pipeline."
	@echo "                  Equivalent to what GitHub Actions runs on push to src/ or tests/."
	@echo ""
	@echo "\033[1;33m▶ CLEANUP\033[0m"
	@echo "  \033[36mclean\033[0m         Remove __pycache__, *.pyc, .coverage, coverage.xml,"
	@echo "                  test-results.xml, .pytest_cache, and htmlcov/."
	@echo ""
	@echo "  \033[36mclean-all\033[0m     Everything in clean, plus removes the .venv/ directory."
	@echo "                  Run 'make install' afterwards to recreate the environment."
	@echo ""
	@echo "─────────────────────────────────────────────────────────────────────────────"
	@echo "  \033[2mFirst time?\033[0m  make install → make env → make notebooks → make serve"
	@echo "─────────────────────────────────────────────────────────────────────────────"
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: venv
venv: ## Create a Python virtual environment in .venv/ (skipped if already exists)
	@if [ ! -d "$(VENV)" ]; then \
		$(PYTHON) -m venv $(VENV); \
		echo "Virtual environment created at $(VENV)/"; \
	else \
		echo "Virtual environment already exists at $(VENV)/ — skipping."; \
	fi

.PHONY: install
install: venv ## Create .venv/ (if needed) and install the project with all dependencies
	$(VENV_PIP) install -e ".[dev]"
	@echo ""
	@echo "Done. Activate the environment with: source $(VENV)/bin/activate"

.PHONY: setup
setup: venv ## Alias for install — create .venv/ and install all dependencies
	$(VENV_PIP) install -e ".[dev]"
	@echo ""
	@echo "Setup complete. Activate the environment with: source $(VENV)/bin/activate"

.PHONY: env
env: ## Create a .env file with placeholder OPENROUTER_API_KEY (skipped if exists)
	@if [ -f .env ]; then \
		echo ".env already exists — skipping. Edit it manually to set OPENROUTER_API_KEY."; \
	else \
		echo "OPENROUTER_API_KEY=sk-or-..." > .env; \
		echo ".env created. Replace the placeholder with your real OpenRouter API key."; \
	fi

# ─────────────────────────────────────────────────────────────────────────────
# Code Quality
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: lint
lint: ## Lint src/ and tests/ with ruff — same check as CI
	ruff check src/ tests/

.PHONY: lint-fix
lint-fix: ## Lint and auto-apply safe fixes with ruff
	ruff check src/ tests/ --fix

# ─────────────────────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: test
test: ## Full test suite with XML coverage report — mirrors CI (target: 80%)
	pytest tests/ -x --cov --cov-report=xml --junitxml=test-results.xml

.PHONY: test-fast
test-fast: ## Run tests without coverage for a faster feedback loop
	pytest tests/ -x

.PHONY: test-verbose
test-verbose: ## Run tests with per-test output and terminal coverage summary
	pytest tests/ -v --cov --cov-report=term-missing

.PHONY: test-features
test-features: ## Run feature engineering tests only (tests/test_features.py)
	pytest tests/test_features.py -v

.PHONY: test-guardrails
test-guardrails: ## Run guardrails tests only — input validation and PII redaction
	pytest tests/test_guardrails.py -v

.PHONY: test-models
test-models: ## Run model training and inference tests only (tests/test_models.py)
	pytest tests/test_models.py -v

# ─────────────────────────────────────────────────────────────────────────────
# Serving
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: serve
serve: ## Start FastAPI server with hot-reload on http://localhost:8000
	uvicorn src.serving.app:app --reload

.PHONY: serve-prod
serve-prod: ## Start FastAPI server in production mode (0.0.0.0:8000, 4 workers)
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --workers 4

# ─────────────────────────────────────────────────────────────────────────────
# Notebooks & Training
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: notebooks
notebooks: ## Open Jupyter in notebooks/ — run 01→02→03→04 to generate model artifacts
	jupyter notebook notebooks/

.PHONY: mlflow
mlflow: ## Open MLflow UI at http://localhost:5000 to browse training experiments
	mlflow ui --backend-store-uri sqlite:///mlflow.db

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: eval-ragas
eval-ragas: ## Run RAGAS pipeline — faithfulness, relevancy, precision, recall
	$(PYTHON) -m evaluation.ragas_eval

.PHONY: eval-judge
eval-judge: ## Run LLM-as-judge evaluation — accuracy, clarity, safety scores
	$(PYTHON) -m evaluation.llm_judge

.PHONY: eval
eval: eval-ragas eval-judge ## Run both evaluation pipelines sequentially (RAGAS + LLM-as-judge)

# ─────────────────────────────────────────────────────────────────────────────
# CI
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: ci
ci: lint test ## Run the full CI pipeline locally: lint then test (same as GitHub Actions)

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: clean
clean: ## Remove Python cache files, test artifacts, and coverage reports
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -f .coverage coverage.xml test-results.xml
	rm -rf .pytest_cache htmlcov

.PHONY: clean-all
clean-all: clean ## Remove everything clean does PLUS the virtual environment
	rm -rf $(VENV)
	@echo "Virtual environment removed. Run 'make setup' to recreate it."
