import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from autoresttest.configurations import OPENAI_LLM_ENGINE, DEFAULT_TEMPERATURE
from autoresttest.prompts.system_prompts import DEFAULT_SYSTEM_MESSAGE
from autoresttest.utils import (
    INPUT_COST_PER_TOKEN,
    OUTPUT_COST_PER_TOKEN,
    encode_dictionary,
)


load_dotenv()


@dataclass
class TokenCounter:
    input_tokens: int = 0
    output_tokens: int = 0


class OpenAILanguageModel:
    # DEPRECATED
    cumulative_cost = 0

    input_tokens = 0
    output_tokens = 0
    cache = {}

    @staticmethod
    # DEPRECATED
    def get_cumulative_cost():
        return OpenAILanguageModel.cumulative_cost

    @staticmethod
    def get_tokens() -> TokenCounter:
        return TokenCounter(
            input_tokens=OpenAILanguageModel.input_tokens,
            output_tokens=OpenAILanguageModel.output_tokens,
        )

    def __init__(
        self, engine=OPENAI_LLM_ENGINE, temperature=DEFAULT_TEMPERATURE, max_tokens=4000
    ):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key is None or self.api_key.strip() == "":
            raise ValueError(
                "OPENAI API key is required for OpenAI language model, found None or empty string."
            )
        self.client = OpenAI(api_key=self.api_key)
        self.engine = engine
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _generate_cache_key(self, user_message, system_message, json_mode):
        key_data = {
            "user_message": user_message,
            "system_message": system_message,
            "json_mode": json_mode,
            "engine": self.engine,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        return encode_dictionary(key_data)

    def query(
        self, user_message, system_message=DEFAULT_SYSTEM_MESSAGE, json_mode=False
    ) -> str:
        cache_key = self._generate_cache_key(user_message, system_message, json_mode)
        if cache_key in OpenAILanguageModel.cache:
            return OpenAILanguageModel.cache[cache_key]

        if json_mode:
            response = self.client.chat.completions.create(
                model=self.engine,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
        else:
            response = self.client.chat.completions.create(
                model=self.engine,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

        input_tokens = (
            response.usage.prompt_tokens
            if hasattr(response.usage, "prompt_tokens")
            else 0
        )
        output_tokens = (
            response.usage.completion_tokens
            if hasattr(response.usage, "completion_tokens")
            else 0
        )
        OpenAILanguageModel.input_tokens += input_tokens
        OpenAILanguageModel.output_tokens += output_tokens
        if self.engine in INPUT_COST_PER_TOKEN:
            OpenAILanguageModel.cumulative_cost += (
                input_tokens * INPUT_COST_PER_TOKEN[self.engine]
            )
        if self.engine in OUTPUT_COST_PER_TOKEN:
            OpenAILanguageModel.cumulative_cost += (
                output_tokens * OUTPUT_COST_PER_TOKEN[self.engine]
            )
        result = response.choices[0].message.content.strip()

        OpenAILanguageModel.cache[cache_key] = result
        return result
