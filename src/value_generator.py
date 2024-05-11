import random
import string
import json
from dataclasses import asdict
from typing import Any, AnyStr, Dict, Iterable

from src.prompts.generator_prompts import (REQUEST_BODY_GEN_PROMPT,
                                           FEWSHOT_REQUEST_BODY_GEN_PROMPT,
                                           PARAMETERS_GEN_PROMPT,
                                           template_gen_prompt,
                                           FEWSHOT_PARAMETER_GEN_PROMPT)
from src.prompts.system_prompts import PARAMETERS_GEN_SYSTEM_MESSAGE, REQUEST_BODY_GEN_SYSTEM_MESSAGE
from src.utils import OpenAILanguageModel
from src.specification_parser import OperationProperties, SchemaProperties, ParameterProperties

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

class SmartValueGenerator:
    def __init__(self, operation_properties: Dict):
        self.operation_properties: Dict = operation_properties
        self.parameters: Dict[str, Dict] = operation_properties.get("parameters")
        self.request_body: Dict[str, Dict] = operation_properties.get("request_body")
        self.summary: str = operation_properties.get("summary")
        self.language_model = OpenAILanguageModel(temperature=0.6)

    def _compose_parameter_gen_prompt(self, GEN_PROMPT, FEWSHOT_PROMPT, schema, is_request_body: bool):
        prompt = f"{GEN_PROMPT}\n{FEWSHOT_PROMPT}\n"
        prompt += template_gen_prompt(summary=self.summary, schema=schema, is_request_body=is_request_body)
        #print("Prompt: ", prompt)
        return prompt

    def _form_parameter_gen_prompt(self, schema: Dict, is_request_body: bool):
        if is_request_body:
            return self._compose_parameter_gen_prompt(REQUEST_BODY_GEN_PROMPT,
                                                     FEWSHOT_REQUEST_BODY_GEN_PROMPT,
                                                     schema,
                                                     is_request_body)
        else:
            return self._compose_parameter_gen_prompt(PARAMETERS_GEN_PROMPT,
                                                      FEWSHOT_PARAMETER_GEN_PROMPT,
                                                     schema,
                                                     is_request_body)

    def _validate_parameters(self, schema: Dict) -> Dict:
        if schema is None:
            return None

        parameters = {}
        for parameter_name, parameter_value in schema.items():
            if parameter_name in self.parameters:
                parameters[parameter_name] = parameter_value
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

    def generate_request_body(self) -> Any:
        """
        Uses the OpenAI language model to generate values for the request body using JSON outputs
        :return: A dictionary of the generated request body
        """
        if self.request_body is None or len(self.request_body) == 0:
            return None

        request_body = {}
        for mime_type, schema in self.request_body.items():
            request_body_prompt = self._form_parameter_gen_prompt(schema, is_request_body=True)
            generated_request_body = self.language_model.query(user_message=request_body_prompt, system_message=REQUEST_BODY_GEN_SYSTEM_MESSAGE, json_mode=True)
            generated_request_body = json.loads(generated_request_body)
            request_body[mime_type] = generated_request_body.get("request_body")
        return request_body


