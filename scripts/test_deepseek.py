import requests
from conv_fin_qa.settings import DEEPSEEK_API_KEY


def test_deepseek():
    """Simple test for calling Deepseek"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": "Hello, world!"}]
    }

    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    print(response.json())


test_deepseek()
