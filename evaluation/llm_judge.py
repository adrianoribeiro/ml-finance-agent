"""LLM-as-judge: evaluate agent responses on 3+ criteria."""
import json
import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from src.agent.react_agent import chat

load_dotenv()
logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are evaluating a credit risk analysis assistant.

Given the user's question, the expected answer, and the agent's actual response,
score the response on these criteria (1-10 each):

1. **Accuracy**: Is the information factually correct? Does it match the expected answer?
2. **Clarity**: Is the response clear and easy to understand?
3. **Completeness**: Does it fully address the question without missing key information?

Respond in JSON format:
{{"accuracy": <score>, "clarity": <score>, "completeness": <score>, "reasoning": "<brief explanation>"}}

Question: {question}
Expected answer: {expected}
Agent response: {response}
"""


def judge_response(question: str, expected: str, response: str) -> dict:
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )

    prompt = JUDGE_PROMPT.format(
        question=question, expected=expected, response=response
    )
    result = llm.invoke(prompt)

    try:
        return json.loads(result.content)
    except json.JSONDecodeError:
        return {"accuracy": 0, "clarity": 0, "completeness": 0, "reasoning": result.content}


def run_judge(golden_set_path: str = "data/golden_set/golden_set.json") -> dict:
    with open(golden_set_path) as f:
        golden_set = json.load(f)

    all_scores = []
    for item in golden_set:
        query = item["query"]
        logger.info(f"Judging: {query}")

        response = chat(query)
        scores = judge_response(query, item["expected_answer"], response)
        scores["question"] = query
        all_scores.append(scores)

        logger.info(f"  accuracy={scores.get('accuracy')} clarity={scores.get('clarity')} completeness={scores.get('completeness')}")

    # Calculate averages
    avg = {
        "accuracy": sum(s.get("accuracy", 0) for s in all_scores) / len(all_scores),
        "clarity": sum(s.get("clarity", 0) for s in all_scores) / len(all_scores),
        "completeness": sum(s.get("completeness", 0) for s in all_scores) / len(all_scores),
        "details": all_scores,
    }

    logger.info(f"Average scores: accuracy={avg['accuracy']:.1f} clarity={avg['clarity']:.1f} completeness={avg['completeness']:.1f}")
    return avg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_judge()
    print("\n=== LLM-as-Judge ===")
    print(f"  Accuracy:     {result['accuracy']:.1f}/10")
    print(f"  Clarity:      {result['clarity']:.1f}/10")
    print(f"  Completeness: {result['completeness']:.1f}/10")
