import time

from .llm_clients import DeepSeekClient, MistralClient


def ask_financial_question(context: str, question, mode: str = "deepseek"):
    """Ask a financial question using the specified model."""
    if mode == "deepseek":
        client = DeepSeekClient()
    elif mode == "mistral":
        client = MistralClient()
    else:
        raise ValueError(f"Mode {mode} not supported")

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
