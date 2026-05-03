"""Application configuration loaded from environment variables."""
import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    # LLM / OpenRouter
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0"))

    # Embedding / RAG
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # File paths
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/credit_model.joblib")
    SCALER_PATH: str = os.getenv("SCALER_PATH", "models/scaler.joblib")
    FEATURE_NAMES_PATH: str = os.getenv("FEATURE_NAMES_PATH", "models/feature_names.joblib")
    DATA_PATH: str = os.getenv("DATA_PATH", "data/processed/credit_risk_clean.csv")
    DOCS_DIR: str = os.getenv("DOCS_DIR", "data/docs")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "models/faiss_index.bin")
    CHUNKS_PATH: str = os.getenv("CHUNKS_PATH", "models/chunks.npy")
    GOLDEN_SET_PATH: str = os.getenv("GOLDEN_SET_PATH", "data/golden_set/golden_set.json")
