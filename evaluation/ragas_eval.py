"""Evaluate RAG pipeline with RAGAS metrics."""
import json
import logging

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)

from src.agent.react_agent import chat
from src.agent.rag_pipeline import RAGRetriever
from src.config import Config

logger = logging.getLogger(__name__)


def load_golden_set(path: str = Config.GOLDEN_SET_PATH) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_evaluation(golden_set_path: str = Config.GOLDEN_SET_PATH) -> dict:
    golden_set = load_golden_set(golden_set_path)
    retriever = RAGRetriever()

    results = []
    for item in golden_set:
        query = item["query"]
        logger.info(f"Evaluating: {query}")

        # Get agent response
        answer = chat(query)

        # Get RAG contexts
        contexts = retriever.search(query, k=3)

        results.append({
            "user_input": query,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": item["expected_answer"],
        })

    dataset = Dataset.from_list(results)

    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        api_key=Config.OPENROUTER_API_KEY,
        base_url=Config.LLM_BASE_URL,
        temperature=Config.LLM_TEMPERATURE,
    )
    evaluator_llm = LangchainLLMWrapper(llm)

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm),
        LLMContextPrecisionWithoutReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
    ]

    scores = evaluate(dataset=dataset, metrics=metrics)

    result = {
        "faithfulness": float(scores["faithfulness"]),
        "answer_relevancy": float(scores["response_relevancy"]),
        "context_precision": float(scores["llm_context_precision_without_reference"]),
        "context_recall": float(scores["context_recall"]),
    }

    logger.info(f"RAGAS scores: {result}")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    scores = run_evaluation()
    print("\n=== RAGAS Evaluation ===")
    for metric, score in scores.items():
        print(f"  {metric}: {score:.4f}")
