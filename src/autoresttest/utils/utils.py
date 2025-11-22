import base64
import hashlib
import json
from typing import Iterable, Dict, List, Any, Optional, Tuple, Set
import itertools
from pathlib import Path

from gensim.downloader import load
import numpy as np

from dotenv import load_dotenv
import os

from autoresttest.config import get_config
from autoresttest.specification import SpecificationParser
from autoresttest.models import ParameterProperties, SchemaProperties
from autoresttest.prompts.generator_prompts import FIX_JSON_OBJ
from autoresttest.prompts.system_prompts import FIX_JSON_SYSTEM_MESSAGE

load_dotenv()
CONFIG = get_config()

CACHE_ROOT = Path(__file__).resolve().parent.parents[2] / "cache"
Q_TABLE_CACHE_DIR = CACHE_ROOT / "q_tables"
GRAPH_CACHE_DIR = CACHE_ROOT / "graphs"


def remove_nulls(item):
    if hasattr(item, 'to_dict'):
        return item.to_dict()
    elif isinstance(item, dict):
        return {k: remove_nulls(v) for k, v in item.items() if v and remove_nulls(v)}
    elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
        return [remove_nulls(i) for i in item if remove_nulls(i) is not None]
    else:
        return item


def get_param_combinations(operation_parameters: Dict[str, ParameterProperties]) -> List[Tuple[str]]:
    param_list = get_params(operation_parameters)
    return get_combinations(param_list)


def get_body_combinations(operation_body: Dict[str, SchemaProperties]) -> Dict[str, List[Tuple[str]]]:
    return {k: get_combinations(v) for k, v in get_request_body_params(operation_body).items()}


def get_body_object_combinations(body_schema: SchemaProperties) -> List[Tuple[str]]:
    return get_combinations(get_body_params(body_schema))


def get_combinations(arr: Iterable[Any]) -> List[Tuple[Any, ...]]:
    """
    Generate combinations from any iterable of parameters.
    """
    arr = list(arr) if arr is not None else []
    combinations = []
    max_size = CONFIG.max_combinations
    # Empirically determined - 16 is max number before size grows too large; configurable for tuning storage size.

    if len(arr) >= max_size:
        for i in range(0, len(arr) - max_size):
            subset = arr[i: i + max_size]
            combinations.extend(
                itertools.chain.from_iterable(
                    itertools.combinations(subset, j) for j in range(1, max_size + 1)
                )
            )
            print(combinations)
        for size in range(max_size + 1, len(arr) + 1):
            for i in range(0, len(arr) - size + 1):
                subset = arr[i: i + size]
                combinations.extend([tuple(subset)])
    else:
        combinations.extend(
            itertools.chain.from_iterable(
                itertools.combinations(arr, i) for i in range(1, len(arr) + 1)
            )
        )

    return combinations


def get_params(operation_parameters: Dict[str, ParameterProperties]) -> List[str]:
    return list(operation_parameters.keys()) if operation_parameters is not None else []


def get_required_params(operation_parameters: Dict[str, ParameterProperties]) -> Set:
    required_parameters = set()
    for parameter, parameter_properties in operation_parameters.items():
        if parameter_properties.required == True:
            required_parameters.add(parameter)
    return required_parameters


def get_required_body_params(operation_body: SchemaProperties) -> Optional[Set]:
    if operation_body is None:
        return None
    required_body = set()

    if operation_body.properties and operation_body.type == "object":
        for key, value in operation_body.properties.items():
            if value.required == True:
                required_body.add(key)

    elif operation_body.items and operation_body.type == "array":
        required_body = get_required_body_params(operation_body.items)

    else:
        return None
    return required_body


def encode_dict_as_key(dictionary: Dict) -> str:
    json_str = json.dumps(dictionary, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def get_body_params(body: SchemaProperties) -> List[str]:
    if body is None:
        return []

    elif body.properties and body.type == "object":
        body_params = []
        for key, value in body.properties.items():
            body_params.append(key)
        return body_params

    elif body.items and body.type == "array":
        return get_body_params(body.items)

    return []


def get_response_params(response: SchemaProperties, response_params: List):
    if response is None:
        return

    if response.properties:
        for key, value in response.properties.items():
            if key not in response_params:
                response_params.append(key)
            get_response_params(value, response_params)

    elif response.items:
        get_response_params(response.items, response_params)


def get_response_param_mappings(response: SchemaProperties, response_mappings):
    if response is None:
        return

    if response.properties:
        for key, value in response.properties.items():
            response_mappings[key] = value
            get_response_param_mappings(value, response_mappings)

    elif response.items:
        get_response_param_mappings(response.items, response_mappings)


def get_request_body_params(operation_body: Dict[str, SchemaProperties]) -> Dict[str, List[str]]:
    return {k: get_body_params(v) for k, v in operation_body.items()} if operation_body is not None else {}


def get_object_shallow_mappings(thing: Any) -> Optional[Dict[str, Any]]:
    """
    Determine the mappings of a given item that contains some nested objects
    :param thing: The thing to get the mappings for
    :return:
    """
    if not thing:
        return None
    mappings = {}
    if type(thing) == dict:
        for key, value in thing.items():
            mappings[key] = value
    elif type(thing) == list and len(thing) > 0:
        mappings = get_object_shallow_mappings(thing[0])
    return mappings


def compose_json_fix_prompt(invalid_json_str: str):
    prompt = FIX_JSON_OBJ
    prompt += invalid_json_str
    return prompt


def attempt_fix_json(invalid_json_str: str):
    from autoresttest.llm import OpenAILanguageModel

    language_model = OpenAILanguageModel(temperature=0.3)
    json_prompt = compose_json_fix_prompt(invalid_json_str)
    fixed_json = language_model.query(user_message=json_prompt, system_message=FIX_JSON_SYSTEM_MESSAGE,
                                      json_mode=True)
    try:
        fixed_json = json.loads(fixed_json)
        return fixed_json
    except json.JSONDecodeError:
        print("Attempt to fix JSON string failed.")
        print(f"Original JSON string: {invalid_json_str}")
        print(f"Fixed JSON string: {fixed_json}")
        return {}


def _is_json_mime(mime_type: str) -> bool:
    """
    Returns True for any JSON-like MIME type.
    """
    if not mime_type:
        return False
    mime_lower = mime_type.lower()
    return (
        "json" in mime_lower
        or mime_lower.endswith("+json")
        or mime_lower.endswith("/json")
    )


def dispatch_request(select_method, full_url: str, params: Dict, body: Dict, header: Optional[Dict] = None):
    """
    Send a request with sensible handling for the provided body and MIME type key (if any)
    """
    params = params or {}
    headers = header if header is not None else {}

    if not body:
        return select_method(full_url, params=params, headers=headers or None)

    if not isinstance(body, dict):
        return select_method(full_url, params=params, data=body, headers=headers or None)

    # Use the first provided MIME type; bodies are expected to be singular.
    mime_type, payload = next(iter(body.items()))
    mime_lower = mime_type.lower() if mime_type else ""

    if _is_json_mime(mime_type):
        headers.setdefault("Content-Type", mime_type)
        return select_method(full_url, params=params, json=payload, headers=headers or None)

    if "x-www-form-urlencoded" in mime_lower:
        headers.setdefault("Content-Type", mime_type)
        body_data = get_object_shallow_mappings(payload)
        if not body_data or not isinstance(body_data, dict):
            body_data = {"data": payload}
        return select_method(full_url, params=params, data=body_data, headers=headers or None)

    if mime_lower.startswith("multipart/"):
        # Let requests set the multipart boundary automatically.
        return select_method(full_url, params=params, files=payload, headers=headers or None)

    if mime_lower.startswith("text/"):
        headers.setdefault("Content-Type", mime_type)
        if not isinstance(payload, str):
            payload = str(payload)
        return select_method(full_url, params=params, data=payload, headers=headers or None)

    # Fallback: send whatever the MIME type is with a best-effort serializer.
    headers.setdefault("Content-Type", mime_type)
    if isinstance(payload, (dict, list)):
        return select_method(full_url, params=params, json=payload, headers=headers or None)
    return select_method(full_url, params=params, data=payload, headers=headers or None)


def encode_dictionary(dictionary) -> str:
    json_str = json.dumps(dictionary, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def is_json_seriable(data):
    try:
        json.dumps(data)
        return True
    except:
        return False


# Pricing for OpenAI API usage as of January 18, 2025

INPUT_COST_PER_TOKEN = {
    "gpt-4o": 2.5e-6,
    "gpt-4o-mini": 0.15e-6,
    "o1": 15e-6,
    "o1-mini": 3e-6
}

OUTPUT_COST_PER_TOKEN = {
    "gpt-4o": 10e-6,
    "gpt-4o-mini": 0.6e-6,
    "o1": 60e-6,
    "o1-mini": 12e-6
}


class EmbeddingModel:
    def __init__(self):
        self.model = load("glove-wiki-gigaword-50")
        self.threshold = 0.8

    def encode_sentence_or_word(self, thing: str):
        words = thing.split(" ")
        word_vectors = [self.model[word] for word in words if word in self.model]
        return np.mean(word_vectors, axis=0) if word_vectors else None

    @staticmethod
    def handle_word_cases(parameter):
        reconstructed_parameter = []
        for index, char in enumerate(parameter):
            if char.isalpha():
                if char.isupper() and index != 0:
                    reconstructed_parameter.append(" " + char.lower())
                elif char == "_" or char == "-":
                    reconstructed_parameter.append(" ")
                else:
                    reconstructed_parameter.append(char)
        return "".join(reconstructed_parameter)


def construct_db_dir():
    for path in (Q_TABLE_CACHE_DIR, GRAPH_CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def get_q_table_cache_path(spec_name: str) -> Path:
    construct_db_dir()
    return Q_TABLE_CACHE_DIR / spec_name


def get_graph_cache_path(spec_name: str) -> Path:
    construct_db_dir()
    return GRAPH_CACHE_DIR / spec_name


def construct_basic_token(token):
    username = token.get("username")
    password = token.get("password")
    token_str = f"{username}:{password}"
    encoded_bytes = base64.b64encode(token_str.encode("utf-8"))
    encoded_str = encoded_bytes.decode("utf-8")
    return f"Basic {encoded_str}"


def get_api_url(spec_parser: SpecificationParser, local_test: bool):
    api_url = spec_parser.get_api_url()
    if not local_test:
        api_url = api_url.replace("localhost", os.getenv("EC2_ADDRESS"))
        api_url = api_url.replace(":9", ":8")
    return api_url
