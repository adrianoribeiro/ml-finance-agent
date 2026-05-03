"""Benchmark agent with different configurations."""
import json
import logging
import time

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.agent.tools import predict_risk, query_data, explain_decision, search_docs
from src.config import Config

logger = logging.getLogger(__name__)

_model = Config.LLM_MODEL

CONFIGS = [
    {"name": f"{_model}-t0", "model": _model, "temperature": 0},
    {"name": f"{_model}-t05", "model": _model, "temperature": 0.5},
    {"name": f"{_model}-t10", "model": _model, "temperature": 1.0},
]

TEST_QUERIES = [
    "Qual a taxa de default no dataset?",
    "Qual o risco de um cliente de 25 anos com renda 3000 e 2 atrasos?",
    "O que significa DebtRatio?",
    "Qual a renda média dos inadimplentes?",
    "Por que um cliente jovem com baixa renda é alto risco?",
]


def run_benchmark() -> list[dict]:
    tools = [predict_risk, query_data, explain_decision, search_docs]
    results = []

    for config in CONFIGS:
        logger.info(f"Testing config: {config['name']}")

        llm = ChatOpenAI(
            model=config["model"],
            api_key=Config.OPENROUTER_API_KEY,
            base_url=Config.LLM_BASE_URL,
            temperature=config["temperature"],
        )

        agent = create_react_agent(
            llm, tools,
            prompt="You are a credit risk analyst assistant. Answer in the same language as the user.",
        )

        config_results = {
            "config": config["name"],
            "model": config["model"],
            "temperature": config["temperature"],
            "queries": [],
            "avg_latency_ms": 0,
        }

        total_time = 0
        for query in TEST_QUERIES:
            start = time.time()
            try:
                result = agent.invoke({"messages": [{"role": "user", "content": query}]})
                response = result["messages"][-1].content
                success = True
            except Exception as e:
                response = str(e)
                success = False

            elapsed = (time.time() - start) * 1000
            total_time += elapsed

            config_results["queries"].append({
                "query": query,
                "response": response[:200],
                "success": success,
                "latency_ms": round(elapsed),
            })

        config_results["avg_latency_ms"] = round(total_time / len(TEST_QUERIES))
        config_results["success_rate"] = sum(
            1 for q in config_results["queries"] if q["success"]
        ) / len(TEST_QUERIES)

        results.append(config_results)
        logger.info(f"  avg_latency={config_results['avg_latency_ms']}ms success_rate={config_results['success_rate']:.0%}")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_benchmark()

    print("\n=== Benchmark Results ===\n")
    for r in results:
        print(f"{r['config']}: latency={r['avg_latency_ms']}ms success={r['success_rate']:.0%}")

    with open("evaluation/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved to evaluation/benchmark_results.json")
