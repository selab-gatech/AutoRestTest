import random
import string
import json
from collections import defaultdict

import requests
from dataclasses import dataclass, field, asdict
from typing import Any, AnyStr, Dict, Optional, List

from autoresttest.config import get_config
from autoresttest.prompts import (
    ENUM_EXAMPLE_CONSTRAINT_PROMPT,
    FAILED_PARAMETER_MATCHINGS_PROMPT,
    FAILED_PARAMETER_RESPONSE_PROMPT,
    FEWSHOT_PARAMETER_GEN_PROMPT,
    FEWSHOT_REQUEST_BODY_GEN_PROMPT,
    FIX_JSON_OBJ,
    FIX_JSON_SYSTEM_MESSAGE,
    IDENTIFY_AUTHENTICATION_GEN_PROMPT,
    IDENTIFY_AUTHENTICATION_SYSTEM_MESSAGE,
    INFORMED_VALUE_AGENT_PROMPT,
    PARAMETER_NECESSITY_PROMPT,
    PARAMETER_REQUIREMENTS_PROMPT,
    PARAMETERS_GEN_PROMPT,
    PARAMETERS_GEN_SYSTEM_MESSAGE,
    REQUEST_BODY_GEN_PROMPT,
    REQUEST_BODY_GEN_SYSTEM_MESSAGE,
    RETRY_PARAMETER_REQUIREMENTS_PROMPT,
    VALUE_AGENT_BODY_FEWSHOT_PROMPT,
    VALUE_AGENT_PARAMS_FEWSHOT_PROMPT,
    get_informed_agent_body_prompt,
    get_informed_agent_params_prompt,
    get_value_agent_body_prompt,
    get_value_agent_params_prompt,
    template_gen_prompt,
)
from autoresttest.utils import (
    remove_nulls,
    get_request_body_params,
    get_object_shallow_mappings,
    attempt_fix_json,
)
from autoresttest.models import (
    OperationProperties,
    ParameterProperties,
    RequestData,
    RequestRequirements,
    RequestResponse,
    SchemaProperties,
)

from .llm import OpenAILanguageModel


CONFIG = get_config()


def randomize_boolean():
    return random.choice([True, False])


def randomize_null():
    return None


def randomize_integer():
    percent = random.randint(1, 100)
    if percent <= 60:
        return random.randint(0, 20)
    elif percent <= 90:
        return random.randint(0, 1000)
    else:
        return random.randint(-(2**10), (2**10))


def randomize_float():
    percent = random.randint(1, 100)
    if percent <= 60:
        return random.uniform(0, 20)
    elif percent <= 90:
        return random.uniform(0, 1000)
    else:
        return random.uniform(-(2**10), (2**10))


def randomize_string():
    percent = random.randint(1, 100)
    if percent <= 60:
        length = random.randint(1, 8)
    elif percent <= 90:
        length = random.randint(4, 20)
    else:
        length = random.randint(1, 50)
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


def randomize_array():
    percent = random.randint(1, 100)
    if percent <= 60:
        length = random.randint(1, 8)
    elif percent <= 90:
        length = random.randint(4, 20)
    else:
        length = random.randint(0, 50)
    return [random.randint(-9999, 9999) for _ in range(length)]


def randomize_object():
    if random.randint(1, 100) <= 90:
        length = random.randint(4, 10)
    else:
        length = random.randint(0, 50)
    return {
        random.choice(string.ascii_letters): random.randint(-9999, 9999)
        for _ in range(length)
    }


def randomized_array_length():
    if random.randint(0, 100) <= 90:
        return random.randint(4, 10)
    else:
        return random.randint(0, 50)


def identify_generator(value: Any):
    generators = {
        "integer": randomize_integer,
        "float": randomize_float,
        "number": randomize_float,
        "boolean": randomize_boolean,
        "string": randomize_string,
        "array": randomize_array,
        "object": randomize_object,
        "null": randomize_null,
    }
    return generators.get(value)


def random_generator():
    generators = {
        "integer": randomize_integer,
        "float": randomize_float,
        "number": randomize_float,
        "boolean": randomize_boolean,
        "string": randomize_string,
        "array": randomize_array,
        "object": randomize_object,
        "null": randomize_null,
    }
    return random.choice(list(generators.values()))


class NaiveValueGenerator:
    def __init__(
        self,
        parameters: Dict[AnyStr, ParameterProperties],
        request_body: Dict[AnyStr, SchemaProperties],
    ):
        self.parameters: Dict[AnyStr, ParameterProperties] = parameters
        self.request_body: Dict[AnyStr, SchemaProperties] = request_body

    def generate_value(self, item_properties: SchemaProperties) -> Any:
        if item_properties.type == "object" or item_properties.properties is not None:
            return {
                item_name: self.generate_value(item_properties)
                for item_name, item_properties in item_properties.properties.items()
            }
        if item_properties.type == "array" or item_properties.items is not None:
            return [
                self.generate_value(item_properties.items)
                for _ in range(randomized_array_length())
            ]
        return (
            identify_generator(item_properties.type)() if item_properties.type else None
        )

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
            request_properties[item_name] = randomized_value  # save diff mime types
        return request_properties


@dataclass
class PromptData:
    GEN_PROMPT: str
    FEWSHOT_PROMPT: str
    schema: Dict
    select_params: Dict = field(default_factory=dict)
    is_request_body: bool = False
    response: requests.Response = None
    failed_mappings: Dict = field(default_factory=dict)


class SmartValueGenerator:
    def __init__(
        self,
        operation_properties: OperationProperties,
        requirements: Optional[RequestRequirements] = None,
        engine="gpt-4o",
        temperature=CONFIG.default_temperature,
    ):
        self.operation_properties: OperationProperties = operation_properties
        self.processed_operation = remove_nulls(asdict(operation_properties))
        self.parameters: Dict[str, Dict] = self.processed_operation.get("parameters")
        self.request_body: Dict[str, Dict] = self.processed_operation.get(
            "request_body"
        )
        self.summary: str = self.processed_operation.get("summary")
        self.language_model = OpenAILanguageModel(temperature=temperature)
        self.request_body_reqs: Dict[str, Any] = (
            requirements.request_body_requirements if requirements else {}
        )
        self.parameters_reqs: Dict[str, Any] = (
            requirements.parameter_requirements if requirements else {}
        )

    def _compose_parameter_gen_prompt(self, prompt_data: PromptData, necessary=False):
        GEN_PROMPT = prompt_data.GEN_PROMPT
        FEWSHOT_PROMPT = prompt_data.FEWSHOT_PROMPT
        schema = prompt_data.schema
        select_params = prompt_data.select_params
        is_request_body = prompt_data.is_request_body

        prompt = f"{GEN_PROMPT}\n"
        prompt += template_gen_prompt(summary=self.summary, schema=schema)

        if necessary:
            prompt += (
                PARAMETER_NECESSITY_PROMPT + "\n".join(select_params.keys()) + "\n\n"
            )
        else:
            prompt += (
                PARAMETER_REQUIREMENTS_PROMPT + "\n".join(select_params.keys()) + "\n\n"
            )

        prompt += "Reminder:\n" + ENUM_EXAMPLE_CONSTRAINT_PROMPT + "\n"

        if FEWSHOT_PROMPT:
            prompt += "Here are some examples of creating values from specifications:\n"
            prompt += FEWSHOT_PROMPT + "\n"
            prompt += "SPECIFICATION:\n" + json.dumps(schema, indent=2) + "\n"

        if is_request_body:
            prompt += "REQUEST_BODY VALUES:\n"
        else:
            prompt += "PARAMETER VALUES:\n"

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
        prompt += (
            RETRY_PARAMETER_REQUIREMENTS_PROMPT
            + "\n".join(select_params.keys())
            + "\n\n"
        )
        prompt += (
            FAILED_PARAMETER_MATCHINGS_PROMPT
            + json.dumps(failed_mappings, indent=2)
            + "\n"
        )
        if response is not None:
            prompt += FAILED_PARAMETER_RESPONSE_PROMPT + response.text + "\n\n"
        if is_request_body:
            prompt += "REQUEST_BODY VALUES:\n"
        else:
            prompt += "PARAMETERS VALUES:\n"
        # print("Prompt: ", prompt)
        return prompt

    def compose_informed_value_prompt(
        self, prompt_data: PromptData, responses: List[RequestResponse]
    ):
        GEN_PROMPT = prompt_data.GEN_PROMPT
        schema = prompt_data.schema
        is_request_body = prompt_data.is_request_body
        few_shot_prompt = prompt_data.FEWSHOT_PROMPT

        prompt = f"{GEN_PROMPT}\n\n"
        prompt += template_gen_prompt(summary=self.summary, schema=schema)
        if is_request_body:
            prompt += get_informed_agent_body_prompt() + "\n"
            for request_response in responses:
                if request_response is not None:
                    prompt += (
                        f"PAST REQUEST BODY: {request_response.request.request_body}\n"
                    )
                    prompt += f"STATUS CODE: {request_response.response.status_code}\n"
                    prompt += f"RESPONSE: {request_response.response.text[:1000]}\n\n"

        else:
            prompt += get_informed_agent_params_prompt() + "\n"
            for request_response in responses:
                if request_response is not None:
                    prompt += (
                        f"PAST PARAMETERS: {request_response.request.parameters}\n"
                    )
                    prompt += f"STATUS CODE: {request_response.response.status_code}\n"
                    prompt += f"RESPONSE: {request_response.response.text[:1000]}\n\n"

        prompt += "Regardless of the past responses:"
        prompt += ENUM_EXAMPLE_CONSTRAINT_PROMPT + "\n"

        prompt += "Here are some examples of creating values from specifications:\n"
        prompt += few_shot_prompt + "\n"

        prompt += "SPECIFICATION:\n" + json.dumps(schema, indent=2) + "\n"

        if is_request_body:
            prompt += "REQUEST_BODY VALUES:\n"
        else:
            prompt += "PARAMETER VALUES:\n"
        return prompt

    def _compose_auth_gen_prompt(self, schema):
        prompt = IDENTIFY_AUTHENTICATION_GEN_PROMPT
        prompt += template_gen_prompt(summary=self.summary, schema=schema) + "\n"
        prompt += "AUTHENTICATION PARAMETERS:\n"
        # print("Prompt: ", prompt)
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

    def _form_parameter_gen_prompt(
        self, schema: Dict, is_request_body: bool, necessary: bool = False
    ):
        if is_request_body:
            prompt_data = PromptData(
                GEN_PROMPT=REQUEST_BODY_GEN_PROMPT,
                FEWSHOT_PROMPT=FEWSHOT_REQUEST_BODY_GEN_PROMPT,
                schema=schema,
                select_params=self._isolate_nonreq_request_body(schema),
                is_request_body=is_request_body,
            )
            return self._compose_parameter_gen_prompt(prompt_data, necessary=necessary)
        else:
            prompt_data = PromptData(
                GEN_PROMPT=PARAMETERS_GEN_PROMPT,
                FEWSHOT_PROMPT=FEWSHOT_PARAMETER_GEN_PROMPT,
                schema=schema,
                select_params=self._isolate_nonreq_params(schema),
                is_request_body=is_request_body,
            )
            return self._compose_parameter_gen_prompt(prompt_data, necessary=necessary)

    def _form_retry_parameter_gen_prompt(
        self,
        schema: Dict,
        failed_mappings: Dict,
        response: requests.Response,
        is_request_body: bool,
    ):
        if is_request_body:
            prompt_data = PromptData(
                GEN_PROMPT=REQUEST_BODY_GEN_PROMPT,
                FEWSHOT_PROMPT=FEWSHOT_REQUEST_BODY_GEN_PROMPT,
                schema=schema,
                select_params=self._isolate_nonreq_request_body(schema),
                is_request_body=is_request_body,
                response=response,
                failed_mappings=failed_mappings,
            )
            return self._compose_retry_parameter_gen_prompt(prompt_data)
        else:
            prompt_data = PromptData(
                GEN_PROMPT=PARAMETERS_GEN_PROMPT,
                FEWSHOT_PROMPT=FEWSHOT_PARAMETER_GEN_PROMPT,
                schema=schema,
                select_params=self._isolate_nonreq_params(schema),
                is_request_body=is_request_body,
                response=response,
                failed_mappings=failed_mappings,
            )
            return self._compose_retry_parameter_gen_prompt(prompt_data)

    def _form_value_agent_prompt(
        self, schema: Dict, is_request_body: bool, num_values: int
    ):
        if is_request_body:
            prompt_data = PromptData(
                GEN_PROMPT=get_value_agent_body_prompt(num_values),
                FEWSHOT_PROMPT=VALUE_AGENT_BODY_FEWSHOT_PROMPT,
                schema=schema,
                select_params=self._isolate_nonreq_request_body(schema),
                is_request_body=is_request_body,
            )
            return self._compose_parameter_gen_prompt(prompt_data, necessary=False)
        else:
            prompt_data = PromptData(
                GEN_PROMPT=get_value_agent_params_prompt(num_values),
                FEWSHOT_PROMPT=VALUE_AGENT_PARAMS_FEWSHOT_PROMPT,
                schema=schema,
                select_params=self._isolate_nonreq_params(schema),
                is_request_body=is_request_body,
            )
            return self._compose_parameter_gen_prompt(prompt_data, necessary=True)

    def _validate_parameters(self, schema: Dict) -> Optional[Dict]:
        if schema is None:
            return None
        parameters = {}
        for parameter_name, parameter_value in schema.items():
            if (
                parameter_name in self.parameters
                and parameter_name not in self.parameters_reqs
            ):
                parameters[parameter_name] = parameter_value
        parameters.update(self.parameters_reqs)
        return parameters

    def generate_parameters(self, necessary=False) -> Optional[Dict[str, Any]]:
        """
        Uses the OpenAI language model to generate values for the parameters using JSON outputs
        :return: A dictionary of the generated parameters
        """
        if self.parameters is None or len(self.parameters) == 0:
            return None

        parameter_prompt = self._form_parameter_gen_prompt(
            schema=self.parameters, is_request_body=False, necessary=necessary
        )
        generated_parameters = self.language_model.query(
            user_message=parameter_prompt,
            system_message=PARAMETERS_GEN_SYSTEM_MESSAGE,
            json_mode=True,
        )
        try:
            generated_parameters = json.loads(generated_parameters)
        except json.JSONDecodeError:
            generated_parameters = attempt_fix_json(generated_parameters)
        parameter_matchings = self._validate_parameters(
            generated_parameters.get("parameters")
        )
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

    def generate_request_body(self, necessary=False) -> Optional[Dict[str, Any]]:
        """
        Uses the OpenAI language model to generate values for the request body using JSON outputs
        :return: A dictionary of the generated request body
        """
        if self.request_body is None or len(self.request_body) == 0:
            return None

        request_body = {}
        for mime_type, schema in self.request_body.items():
            request_body_prompt = self._form_parameter_gen_prompt(
                schema=schema, is_request_body=True, necessary=necessary
            )
            generated_request_body = self.language_model.query(
                user_message=request_body_prompt,
                system_message=REQUEST_BODY_GEN_SYSTEM_MESSAGE,
                json_mode=True,
            )
            try:
                generated_request_body = json.loads(generated_request_body)
            except json.JSONDecodeError:
                generated_request_body = attempt_fix_json(generated_request_body)
            validated_request_body = self.validate_request_body(
                generated_request_body.get("request_body")
            )
            request_body[mime_type] = validated_request_body
        return request_body

    def generate_retry_parameters(
        self, failed_request_data: RequestData, response: requests.Response
    ):
        """
        Uses the OpenAI language model to generate values for the parameters using JSON outputs
        :return: A dictionary of the generated parameters
        """
        if self.parameters is None or len(self.parameters) == 0:
            return None

        parameter_prompt = self._form_retry_parameter_gen_prompt(
            schema=self.parameters,
            failed_mappings=failed_request_data.parameters,
            response=response,
            is_request_body=False,
        )
        generated_parameters = self.language_model.query(
            user_message=parameter_prompt,
            system_message=PARAMETERS_GEN_SYSTEM_MESSAGE,
            json_mode=True,
        )
        try:
            generated_parameters = json.loads(generated_parameters)
        except json.JSONDecodeError:
            generated_parameters = attempt_fix_json(generated_parameters)
        parameter_matchings = self._validate_parameters(
            generated_parameters.get("parameters")
        )
        return parameter_matchings

    def generate_retry_request_body(
        self, failed_request_data: RequestData, response: requests.Response
    ):
        """
        Uses the OpenAI language model to generate values for the request body using JSON outputs
        :return: A dictionary of the generated request body
        """
        if self.request_body is None or len(self.request_body) == 0:
            return None

        request_body = {}
        for mime_type, schema in self.request_body.items():
            request_body_prompt = self._form_retry_parameter_gen_prompt(
                schema=schema,
                failed_mappings=failed_request_data.request_body.get(mime_type),
                response=response,
                is_request_body=True,
            )
            generated_request_body = self.language_model.query(
                user_message=request_body_prompt,
                system_message=REQUEST_BODY_GEN_SYSTEM_MESSAGE,
                json_mode=True,
            )
            try:
                generated_request_body = json.loads(generated_request_body)
            except json.JSONDecodeError:
                generated_request_body = attempt_fix_json(generated_request_body)
            validated_request_body = self.validate_request_body(
                generated_request_body.get("request_body")
            )
            request_body[mime_type] = validated_request_body
        return request_body

    def determine_auth_params(self):
        """
        Determines if the operation consists of any authentication information sent as parameters in either the query or the request body
        :return:
        """
        auth_prompt = self._compose_auth_gen_prompt(self.processed_operation)
        auth_parameters = self.language_model.query(
            user_message=auth_prompt,
            system_message=IDENTIFY_AUTHENTICATION_SYSTEM_MESSAGE,
            json_mode=True,
        )
        try:
            auth_parameters = json.loads(auth_parameters)
        except json.JSONDecodeError:
            auth_parameters = attempt_fix_json(auth_parameters)
        return auth_parameters.get("authentication_parameters")

    def _validate_value_params(self, schema: Dict) -> Dict[Any, List]:
        if schema is None:
            return {}
        param_mappings = defaultdict(list)
        for param_name, param_values in schema.items():
            if param_name in self.parameters:
                for param_value in param_values.values():
                    param_mappings[param_name].append(param_value)
        return param_mappings

    def generate_value_agent_params(self, num_values: int) -> Optional[Dict[str, List]]:
        """

        :param num_values:
        :return: A LIST of parameter mappings (dicts) for the operation; should have num_values items in list where each list has the parameter mappings
        """
        if self.parameters is None or len(self.parameters) == 0:
            return {}

        parameter_prompt = self._form_value_agent_prompt(
            schema=self.parameters, is_request_body=False, num_values=num_values
        )
        generated_parameters = self.language_model.query(
            user_message=parameter_prompt,
            system_message=PARAMETERS_GEN_SYSTEM_MESSAGE,
            json_mode=True,
        )
        try:
            generated_parameters = json.loads(generated_parameters)
        except json.JSONDecodeError:
            generated_parameters = attempt_fix_json(generated_parameters)
        parameter_matchings = self._validate_value_params(
            generated_parameters.get("parameters")
        )
        return parameter_matchings

    def _validate_value_body(self, schema: Dict) -> List:
        if schema is None:
            return []
        values = [body for body in schema.values()]
        return values

    def generate_value_agent_body(self, num_values: int) -> Optional[Dict[str, List]]:
        """

        :param num_values:
        :return: A LIST of request body mappings (dicts) for the operation; should have num_values items in list where each list has the request body mappings
        """
        if self.request_body is None or len(self.request_body) == 0:
            return {}

        request_body = {}
        for mime_type, schema in self.request_body.items():
            request_body_prompt = self._form_value_agent_prompt(
                schema=schema, is_request_body=True, num_values=num_values
            )
            generated_request_body = self.language_model.query(
                user_message=request_body_prompt,
                system_message=REQUEST_BODY_GEN_SYSTEM_MESSAGE,
                json_mode=True,
            )
            try:
                generated_request_body = json.loads(generated_request_body)
            except json.JSONDecodeError:
                generated_request_body = attempt_fix_json(generated_request_body)
            validated_request_body = self._validate_value_body(
                generated_request_body.get("request_body")
            )
            request_body[mime_type] = validated_request_body
        return request_body

    def generate_informed_value_agent_body(
        self, num_values: int, responses: List[RequestResponse]
    ):
        if self.request_body is None or len(self.request_body) == 0:
            return {}

        request_body = {}
        for mime_type, schema in self.request_body.items():
            prompt_data = PromptData(
                GEN_PROMPT=get_value_agent_body_prompt(num_values),
                FEWSHOT_PROMPT=VALUE_AGENT_BODY_FEWSHOT_PROMPT,
                schema=schema,
                select_params=self._isolate_nonreq_request_body(schema),
                is_request_body=True,
            )
            request_body_prompt = self.compose_informed_value_prompt(
                prompt_data, responses
            )
            generated_request_body = self.language_model.query(
                user_message=request_body_prompt,
                system_message=REQUEST_BODY_GEN_SYSTEM_MESSAGE,
                json_mode=True,
            )
            try:
                generated_request_body = json.loads(generated_request_body)
            except json.JSONDecodeError:
                generated_request_body = attempt_fix_json(generated_request_body)
            validated_request_body = self._validate_value_body(
                generated_request_body.get("request_body")
            )
            request_body[mime_type] = validated_request_body
        return request_body

    def generate_informed_value_agent_params(
        self, num_values: int, responses: List[RequestResponse]
    ):
        if self.parameters is None or len(self.parameters) == 0:
            return {}

        prompt_data = PromptData(
            GEN_PROMPT=get_value_agent_params_prompt(num_values),
            FEWSHOT_PROMPT=VALUE_AGENT_PARAMS_FEWSHOT_PROMPT,
            schema=self.parameters,
            select_params=self._isolate_nonreq_params(self.parameters),
            is_request_body=False,
        )
        parameter_prompt = self.compose_informed_value_prompt(prompt_data, responses)
        generated_parameters = self.language_model.query(
            user_message=parameter_prompt,
            system_message=PARAMETERS_GEN_SYSTEM_MESSAGE,
            json_mode=True,
        )
        try:
            generated_parameters = json.loads(generated_parameters)
        except json.JSONDecodeError:
            generated_parameters = attempt_fix_json(generated_parameters)
        parameter_matchings = self._validate_value_params(
            generated_parameters.get("parameters")
        )
        return parameter_matchings
