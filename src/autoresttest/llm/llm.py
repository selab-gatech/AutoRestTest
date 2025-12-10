import os
import threading
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI

from autoresttest.config import get_config
from autoresttest.prompts.system_prompts import DEFAULT_SYSTEM_MESSAGE
from autoresttest.utils import (
    INPUT_COST_PER_TOKEN,
    OUTPUT_COST_PER_TOKEN,
    encode_dictionary,
)

CONFIG = get_config()

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

    # Thread-safety locks for parallel value generation
    _cache_lock = threading.RLock()
    _token_lock = threading.RLock()

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
        self,
        engine=CONFIG.openai_llm_engine,
        temperature=CONFIG.creative_temperature,
        max_tokens=20000,
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

    def _max_tokens_field(self) -> str:
        """
        Prefer the newer OpenAI interface (max_completion_tokens). Fall back to max_tokens
        for older model families that still expect it.
        """
        engine_str = str(self.engine)
        if engine_str.startswith(("gpt-3.5", "gpt-4", "gpt-4o")):
            return "max_tokens"
        return "max_completion_tokens"

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

        # Thread-safe cache read
        with OpenAILanguageModel._cache_lock:
            if cache_key in OpenAILanguageModel.cache:
                return OpenAILanguageModel.cache[cache_key]

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        use_legacy_max_tokens = self._max_tokens_field() == "max_tokens"

        if json_mode:
            if use_legacy_max_tokens:
                response = self.client.chat.completions.create(
                    model=self.engine,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.engine,
                    messages=messages,
                    max_completion_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
        else:
            if use_legacy_max_tokens:
                response = self.client.chat.completions.create(
                    model=self.engine,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.engine,
                    messages=messages,
                    max_completion_tokens=self.max_tokens,
                    temperature=self.temperature,
                )

        input_tokens = 0
        output_tokens = 0
        if response.usage is not None:
            input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(response.usage, "completion_tokens", 0) or 0

        # Thread-safe token and cost updates
        with OpenAILanguageModel._token_lock:
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

        if not response.choices:
            return ""
        content = response.choices[0].message.content
        result = content.strip() if content else ""

        # Thread-safe cache write
        with OpenAILanguageModel._cache_lock:
            OpenAILanguageModel.cache[cache_key] = result

        return result
