import random
import string
import json
import requests
from dataclasses import asdict, dataclass
from typing import Any, AnyStr, Dict, Iterable, TYPE_CHECKING

from src.prompts.generator_prompts import (REQUEST_BODY_GEN_PROMPT,
                                           FEWSHOT_REQUEST_BODY_GEN_PROMPT,
                                           PARAMETERS_GEN_PROMPT,
                                           template_gen_prompt,
                                           FEWSHOT_PARAMETER_GEN_PROMPT,
                                           PARAMETER_REQUIREMENTS_PROMPT, RETRY_PARAMETER_REQUIREMENTS_PROMPT,
                                           FAILED_PARAMETER_MATCHINGS_PROMPT, FAILED_PARAMETER_RESPONSE_PROMPT)
from src.prompts.system_prompts import PARAMETERS_GEN_SYSTEM_MESSAGE, REQUEST_BODY_GEN_SYSTEM_MESSAGE
from src.utils import OpenAILanguageModel
from src.specification_parser import OperationProperties, SchemaProperties, ParameterProperties

if TYPE_CHECKING:
    from src.request_generator import RequestRequirements, RequestData

def randomize_boolean():
    return random.choice([True, False])

def randomize_null():
    return None

def randomize_integer():
    if random.randint(1, 100) <= 70:
        return random.randint(0, 1000)
    else:
        return random.randint(-(2 ** 10), (2 ** 10))

def randomize_float():
    if random.randint(1, 100) <= 70:
        return random.uniform(0, 1000)
    else:
        return random.uniform(-(2 ** 10), (2 ** 10))

def randomize_string():
    if random.randint(1, 100) <= 90:
        length = random.randint(4, 16)
    else:
        length = random.randint(1, 100)
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def randomize_array():
    if random.randint(1, 100) <= 90:
        length = random.randint(4, 10)
    else:
        length = random.randint(0, 50)
    return [random.randint(-9999, 9999) for _ in range(length)]

def randomize_object():
    if random.randint(1, 100) <= 90:
        length = random.randint(4, 10)
    else:
        length = random.randint(0, 50)
    return {random.choice(string.ascii_letters): random.randint(-9999, 9999) for _ in range(length)}

def randomized_array_length():
    if random.randint(0, 100) <= 90:
        return random.randint(4, 10)
    else:
        return random.randint(0, 50)

def identify_generator(value: Any):
    generators = {"integer": randomize_integer,
                  "float": randomize_float,
                  "number": randomize_float,
                  "boolean": randomize_boolean,
                  "string": randomize_string,
                  "array": randomize_array,
                  "object": randomize_object,
                  "null": randomize_null}
    return generators.get(value)

class NaiveValueGenerator:
    def __init__(self, parameters: Dict[AnyStr, ParameterProperties], request_body: Dict[AnyStr, SchemaProperties]):
        self.parameters: Dict[AnyStr, ParameterProperties] = parameters
        self.request_body: Dict[AnyStr, SchemaProperties] = request_body

    def generate_value(self, item_properties: SchemaProperties) -> Any:
        if item_properties.type == "object" or item_properties.properties is not None:
            return {item_name: self.generate_value(item_properties) for item_name, item_properties in item_properties.properties.items()}
        if item_properties.type == "array" or item_properties.items is not None:
            return [self.generate_value(item_properties.items) for _ in range(randomized_array_length())]
        return identify_generator(item_properties.type)() if item_properties.type else None

    def generate_parameters(self) -> Dict[AnyStr, Any]:
        query_parameters = {}
        for parameter_name, parameter_properties in self.parameters.items():
            randomized_value = self.generate_value(parameter_properties.schema)
            query_parameters[parameter_name] = randomized_value
        return query_parameters

    def generate_request_body(self):
        request_properties = {}
        for item_name, item_properties in self.request_body.items():
            randomized_value = self.generate_value(item_properties)
            request_properties[item_name] = randomized_value # save diff mime types
        return request_properties

@dataclass
class PromptData:
    GEN_PROMPT: str
    FEWSHOT_PROMPT: str
    schema: Dict
    select_params: Dict
    is_request_body: bool
    response: requests.Response = None
    failed_mappings: Dict = None

class SmartValueGenerator:
    def __init__(self, operation_properties: Dict, requirements: 'RequestRequirements' = None, engine="gpt-3.5-turbo-0125"):
        self.operation_properties: Dict = operation_properties
        self.parameters: Dict[str, Dict] = operation_properties.get("parameters")
        self.request_body: Dict[str, Dict] = operation_properties.get("request_body")
        self.summary: str = operation_properties.get("summary")
        self.language_model = OpenAILanguageModel(engine=engine,temperature=0.6)
        self.request_body_reqs: Dict[str, Any] = requirements.request_body_requirements if requirements else {}
        self.parameters_reqs: Dict[str, Any] = requirements.parameter_requirements if requirements else {}

    def _compose_parameter_gen_prompt(self, prompt_data: PromptData):
        GEN_PROMPT = prompt_data.GEN_PROMPT
        FEWSHOT_PROMPT = prompt_data.FEWSHOT_PROMPT
        schema = prompt_data.schema
        select_params = prompt_data.select_params
        is_request_body = prompt_data.is_request_body

        prompt = f"{GEN_PROMPT}\n{FEWSHOT_PROMPT}\n"
        prompt += template_gen_prompt(summary=self.summary, schema=schema)
        prompt += PARAMETER_REQUIREMENTS_PROMPT + '\n'.join(select_params.keys()) + "\n\n"
        if is_request_body:
            prompt += "REQUEST_BODY:\n"
        else:
            prompt += "PARAMETERS:\n"
        print("Prompt: ", prompt)
        return prompt

    def _compose_retry_parameter_gen_prompt(self, prompt_data: PromptData):
        GEN_PROMPT = prompt_data.GEN_PROMPT
        FEWSHOT_PROMPT = prompt_data.FEWSHOT_PROMPT
        schema = prompt_data.schema
        select_params = prompt_data.select_params
        is_request_body = prompt_data.is_request_body
        response = prompt_data.response
        failed_mappings = prompt_data.failed_mappings

        prompt = f"{GEN_PROMPT}\n{FEWSHOT_PROMPT}\n"
        prompt += template_gen_prompt(summary=self.summary, schema=schema)
        prompt += RETRY_PARAMETER_REQUIREMENTS_PROMPT + '\n'.join(select_params.keys()) + "\n\n"
        prompt += FAILED_PARAMETER_MATCHINGS_PROMPT + json.dumps(failed_mappings, indent=2) + "\n"
        prompt += FAILED_PARAMETER_RESPONSE_PROMPT + response.text + "\n\n"
        if is_request_body:
            prompt += "REQUEST_BODY:\n"
        else:
            prompt += "PARAMETERS:\n"
        print("Prompt: ", prompt)
        return prompt

    def _isolate_nonreq_params(self, schema: Dict[str, Dict], is_request_body=False):
        if type(schema) is not dict:
            return {}
        nonreq_params = {}
        for param_name, param_properties in schema.items():
            if not is_request_body and param_name not in self.parameters_reqs:
                nonreq_params[param_name] = param_properties
            if is_request_body and param_name not in self.request_body_reqs:
                nonreq_params[param_name] = param_properties
        return nonreq_params

    def _isolate_nonreq_request_body(self, schema: Dict):
        if schema.get("properties"):
            # NOTE: We do not handle nested objects
            nonreq_request_body = self._isolate_nonreq_params(schema.get("properties"))
        elif schema.get("items"):
            nonreq_request_body = self._isolate_nonreq_request_body(schema.get("items"))
        else:
            nonreq_request_body = self._isolate_nonreq_params(schema)
        return nonreq_request_body

    def _form_parameter_gen_prompt(self, schema: Dict, is_request_body: bool):
        if is_request_body:
            prompt_data = PromptData(GEN_PROMPT=REQUEST_BODY_GEN_PROMPT,
                                        FEWSHOT_PROMPT=FEWSHOT_REQUEST_BODY_GEN_PROMPT,
                                        schema=schema,
                                        select_params=self._isolate_nonreq_request_body(schema),
                                        is_request_body=is_request_body)
            return self._compose_parameter_gen_prompt(prompt_data)
        else:
            prompt_data = PromptData(GEN_PROMPT=PARAMETERS_GEN_PROMPT,
                                        FEWSHOT_PROMPT=FEWSHOT_PARAMETER_GEN_PROMPT,
                                        schema=schema,
                                        select_params=self._isolate_nonreq_params(schema),
                                        is_request_body=is_request_body)
            return self._compose_parameter_gen_prompt(prompt_data)

    def _form_retry_parameter_gen_prompt(self, schema: Dict, failed_mappings: Dict, response: requests.Response, is_request_body: bool):
        if is_request_body:
            prompt_data = PromptData(GEN_PROMPT=REQUEST_BODY_GEN_PROMPT,
                                        FEWSHOT_PROMPT=FEWSHOT_REQUEST_BODY_GEN_PROMPT,
                                        schema=schema,
                                        select_params=self._isolate_nonreq_request_body(schema),
                                        is_request_body=is_request_body,
                                        response=response,
                                        failed_mappings=failed_mappings)
            return self._compose_retry_parameter_gen_prompt(prompt_data)
        else:
            prompt_data = PromptData(GEN_PROMPT=PARAMETERS_GEN_PROMPT,
                                        FEWSHOT_PROMPT=FEWSHOT_PARAMETER_GEN_PROMPT,
                                        schema=schema,
                                        select_params=self._isolate_nonreq_params(schema),
                                        is_request_body=is_request_body,
                                        response=response,
                                        failed_mappings=failed_mappings)
            return self._compose_retry_parameter_gen_prompt(prompt_data)

    def _validate_parameters(self, schema: Dict) -> Dict:
        if schema is None:
            return None
        parameters = {}
        for parameter_name, parameter_value in schema.items():
            if parameter_name in self.parameters and parameter_name not in self.parameters_reqs:
                parameters[parameter_name] = parameter_value
        parameters.update(self.parameters_reqs)
        return parameters

    def generate_parameters(self) -> Dict[str, Any]:
        """
        Uses the OpenAI language model to generate values for the parameters using JSON outputs
        :return: A dictionary of the generated parameters
        """
        if self.parameters is None or len(self.parameters) == 0:
            return None

        parameter_prompt = self._form_parameter_gen_prompt(schema=self.parameters, is_request_body=False)
        generated_parameters = self.language_model.query(user_message=parameter_prompt, system_message=PARAMETERS_GEN_SYSTEM_MESSAGE, json_mode=True)
        generated_parameters = json.loads(generated_parameters)
        parameter_matchings = self._validate_parameters(generated_parameters.get("parameters"))
        return parameter_matchings

    def validate_request_body(self, schema: Any):
        if schema is None:
            return None
        if type(schema) is dict:
            # NOTE: We do not handle nested objects
            schema.update(self.request_body_reqs)
            return schema
        elif type(schema) is list:
            for i in range(len(schema)):
                schema[i] = self.validate_request_body(schema[i])
        return schema

    def generate_request_body(self) -> Dict[str, Any]:
        """
        Uses the OpenAI language model to generate values for the request body using JSON outputs
        :return: A dictionary of the generated request body
        """
        if self.request_body is None or len(self.request_body) == 0:
            return None

        request_body = {}
        for mime_type, schema in self.request_body.items():
            request_body_prompt = self._form_parameter_gen_prompt(schema=schema, is_request_body=True)
            generated_request_body = self.language_model.query(user_message=request_body_prompt, system_message=REQUEST_BODY_GEN_SYSTEM_MESSAGE, json_mode=True)
            generated_request_body = json.loads(generated_request_body)
            validated_request_body = self.validate_request_body(generated_request_body.get("request_body"))
            request_body[mime_type] = validated_request_body
        return request_body

    def generate_retry_parameters(self, failed_request_data: 'RequestData', response: requests.Response):
        """
        Uses the OpenAI language model to generate values for the parameters using JSON outputs
        :return: A dictionary of the generated parameters
        """
        if self.parameters is None or len(self.parameters) == 0:
            return None

        parameter_prompt = self._form_retry_parameter_gen_prompt(schema=self.parameters, failed_mappings=failed_request_data.parameters, response=response, is_request_body=False)
        generated_parameters = self.language_model.query(user_message=parameter_prompt, system_message=PARAMETERS_GEN_SYSTEM_MESSAGE, json_mode=True)
        generated_parameters = json.loads(generated_parameters)
        parameter_matchings = self._validate_parameters(generated_parameters.get("parameters"))
        return parameter_matchings

    def generate_retry_request_body(self, failed_request_data: 'RequestData', response: requests.Response):
        """
        Uses the OpenAI language model to generate values for the request body using JSON outputs
        :return: A dictionary of the generated request body
        """
        if self.request_body is None or len(self.request_body) == 0:
            return None

        request_body = {}
        for mime_type, schema in self.request_body.items():
            request_body_prompt = self._form_retry_parameter_gen_prompt(schema=schema, failed_mappings=failed_request_data.request_body.get(mime_type), response=response, is_request_body=True)
            generated_request_body = self.language_model.query(user_message=request_body_prompt, system_message=REQUEST_BODY_GEN_SYSTEM_MESSAGE, json_mode=True)
            generated_request_body = json.loads(generated_request_body)
            validated_request_body = self.validate_request_body(generated_request_body.get("request_body"))
            request_body[mime_type] = validated_request_body
        return request_body


