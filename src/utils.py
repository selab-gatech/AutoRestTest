from typing import Iterable

from openai import OpenAI
from dotenv import load_dotenv
import os

from src.prompts.system_prompts import DEFAULT_SYSTEM_MESSAGE

load_dotenv()

def remove_nulls(item):
    if hasattr(item, 'to_dict'):
        return item.to_dict()
    elif isinstance(item, dict):
        return {k: remove_nulls(v) for k, v in item.items() if v is not None and remove_nulls(v) is not None}
    elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
        return [remove_nulls(i) for i in item if remove_nulls(i) is not None]
    else:
        return item

# OpenAI available engines = ["gpt-3.5-turbo-0125", "gpt-4-turbo"]

class OpenAILanguageModel:
    def __init__(self, engine = "gpt-3.5-turbo-0125", temperature = 0.8, max_tokens = 1000):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key is None or self.api_key.strip() == "":
            raise ValueError("OPENAI API key is required for OpenAI language model, found None or empty string.")
        self.client = OpenAI(api_key=self.api_key)
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens

    def query(self, user_message, system_message = DEFAULT_SYSTEM_MESSAGE, json_mode = False) -> str:
        if json_mode:
            response = self.client.chat.completions.create(
                model=self.engine,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type":"json_object"}
            )
        else:
            response = self.client.chat.completions.create(
                model=self.engine,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
        return response.choices[0].message.content.strip()




