"""Configuration loading utilities for AutoRestTest."""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
import tomli as tomllib

from pydantic import BaseModel, ConfigDict


CONFIG_FILE_NAME = "configurations.toml"
CONFIG_DIR = Path(__file__).resolve().parent
AUTORESTTEST_DIR = CONFIG_DIR.parent
PROJECT_ROOT = AUTORESTTEST_DIR.parent.parent
CONFIG_PATH = PROJECT_ROOT / CONFIG_FILE_NAME


class SpecConfig(BaseModel):
    location: str
    recursion_limit: int = 50
    strict_validation: bool = True


class LLMConfig(BaseModel):
    engine: str
    temperature: float


class HeaderAgentConfig(BaseModel):
    enabled: bool


class AgentsConfig(BaseModel):
    header: HeaderAgentConfig


class AgentCombinationConfig(BaseModel):
    max_combinations: int = 10


class CacheConfig(BaseModel):
    use_cached_graph: bool
    use_cached_table: bool


class QLearningConfig(BaseModel):
    learning_rate: float
    discount_factor: float
    max_exploration: float


class RequestGenerationConfig(BaseModel):
    time_duration: int
    mutation_rate: float


class Config(BaseModel):
    spec: SpecConfig
    llm: LLMConfig
    agents: AgentsConfig
    agent: AgentCombinationConfig
    cache: CacheConfig
    q_learning: QLearningConfig
    request_generation: RequestGenerationConfig

    model_config = ConfigDict(frozen=True)

    @property
    def specification_location(self) -> str:
        return self.spec.location

    @property
    def recursion_limit(self) -> int:
        return self.spec.recursion_limit

    @property
    def strict_validation(self) -> bool:
        return self.spec.strict_validation

    @property
    def openai_llm_engine(self) -> str:
        return self.llm.engine

    @property
    def default_temperature(self) -> float:
        return self.llm.temperature

    @property
    def enable_header_agent(self) -> bool:
        return self.agents.header.enabled

    @property
    def max_combinations(self) -> int:
        return self.agent.max_combinations


def _load_raw_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_PATH}")
    with CONFIG_PATH.open("rb") as fh:
        return tomllib.load(fh)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the cached configuration values."""
    return Config.model_validate(_load_raw_config())


__all__ = [
    "Config",
    "CONFIG_PATH",
    "PROJECT_ROOT",
    "get_config",
]
