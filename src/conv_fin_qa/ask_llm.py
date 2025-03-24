import time

from enum import Enum

from .llm_clients import DeepSeekClient, MistralClient


class UsageMode(Enum):
    USER = 0
    EVALUATION = 1
    TRAINING = 2


def ask_financial_question(context: str, question, model: str = "deepseek"):
    """Ask a financial question using the specified model."""
    if model == "deepseek":
        client = DeepSeekClient()
    elif model == "mistral":
        client = MistralClient()
    else:
        raise ValueError(f"Model {model} not supported")

    start_time = time.time()
    response = client.get_response(
        context=context,
        question=question
    )
    print(f"\nTime taken: {time.time() - start_time:.2f}s")
    print(f"""
Question: {question}
Answer: {response}""")
    return response
