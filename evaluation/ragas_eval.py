"""Evaluate RAG pipeline with RAGAS metrics."""
import json
import logging
import os

from datasets import Dataset
from dotenv import load_dotenv
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

load_dotenv()
logger = logging.getLogger(__name__)


def load_golden_set(path: str = "data/golden_set/golden_set.json") -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_evaluation(golden_set_path: str = "data/golden_set/golden_set.json") -> dict:
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
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
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
