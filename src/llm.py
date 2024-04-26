from openai import OpenAI
from dotenv import load_dotenv
from .classification_prompts import *
import os

load_dotenv()

class OpenAILanguageModel:
    def __init__(self, engine = "gpt-4-turbo-preview", temperature = 0.8, max_tokens = 1000):
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




