import logging
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from src.agent.tools import predict_risk, query_data, explain_decision, search_docs

load_dotenv()
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a credit risk analyst assistant. "
    "You help users evaluate client default risk using real data and ML models. "
    "Always be objective and base your answers on the tools available. "
    "Answer in the same language as the user's question."
)


def create_agent():
    llm = ChatOpenAI(
        model="openai/gpt-4o-mini",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        temperature=0,
    )

    tools = [predict_risk, query_data, explain_decision, search_docs]

    return create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)


def chat(query: str) -> str:
    agent = create_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    return result["messages"][-1].content
