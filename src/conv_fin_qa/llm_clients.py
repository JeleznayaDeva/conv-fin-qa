from abc import ABC, abstractmethod
from pathlib import Path
import yaml
from functools import cached_property
from typing import Optional
from llama_cpp import Llama
import requests

import logging
from .settings import MODEL_PATH, CLIENTS_PATH, DEEPSEEK_API_KEY

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract class for interacting with LLM clients."""
    @abstractmethod
    def prompt(self, context: str, question: str) -> str:
        pass

    @abstractmethod
    def get_response(self, context: str, question: str) -> dict:
        pass

    @cached_property
    def system_dict(self):
        return self.load_client_config(Path(CLIENTS_PATH))

    @cached_property
    def system(self) -> str:
        return self.system_dict["system"]

    def load_client_config(self, path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist")
        return yaml.safe_load(path.read_text())


class MistralClient(LLMClient):
    """Main class for interacting with Mistal loaded locally."""
    def __init__(self) -> None:
        self.llm = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,  # Context window
            n_threads=8  # Use all CPU cores
        )

    @cached_property
    def examples(self) -> str:
        return "\n".join(f"{key}:\n{value}" for example in self.system_dict["examples"] for key, value in example.items())

    def prompt(self, context: str, question: str) -> str:
        return f"""[INST] <<SYS>>
        {self.system}
        <</SYS>>
        Examples:
        {self.examples}

        question: {question}
        context: {context} [/INST]"""

    def get_response(self, context: str, question: str) -> dict:
        try:
            return self.llm(
                self.prompt(context, question),
                max_tokens=256,
                stop=["</s>"],
                temperature=0.1
            )['choices'][0]['text']
        except Exception as e:
            logger.warning(f"{e}. Skipping this response.")
            return None


class DeepSeekClient(LLMClient):
    """Main class for interacting with DeepSeek API"""
    def __init__(self) -> None:
        self.headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

    def prompt(self, context: str, question: str) -> str:
        prompt = f"<s>[INST] <<SYS>>\n{self.system}\n<</SYS>>\n\n"

        # Add examples
        for example in self.system_dict["examples"]:
            prompt += f"""
Context:\n{example['context']}\n\nQuestion: {example['question']} [/INST]\n
{example['response']}\n</s><s>[INST]
"""

        # Add current query
        prompt += f"Context:\n{context}\n\nQuestion: {question} [/INST]"
        return prompt

    def get_response(self, context: str, question: str) -> Optional[dict]:
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": self.prompt(context, question)}],
            "temperature": 0.1,
            "max_tokens": 100
        }
        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content'].strip()

        except Exception as e:
            logger.warning(f"{e}. Skipping this response.")
            return None
