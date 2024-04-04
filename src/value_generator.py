import random
import string
from typing import Any, AnyStr, Dict

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
        if item_properties.type == "object":
            return {item_name: self.generate_value(item_properties) for item_name, item_properties in item_properties.properties.items()}
        if item_properties.type == "array":
            return [self.generate_value(item_properties.items) for _ in range(randomized_array_length())]
        return identify_generator(item_properties.type)()

    #def generate_values(self, parameters: Dict[AnyStr, ParameterProperties], request_body: Dict[AnyStr, SchemaProperties]):
    ##    parameter_mappings = {}
    #    request_body = None
    #    for parameter in parameters.items()

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
class InformedValueGenerator: 
    def __init__(self, parameters: Dict[AnyStr, ParameterProperties], request_body: Dict[AnyStr, SchemaProperties]):
        self.parameters: Dict[AnyStr, ParameterProperties] = parameters
        self.request_body: Dict[AnyStr, SchemaProperties] = request_body
    def generate_value(self, item_properties: SchemaProperties) -> Any:
        pass 
    def generate_parameters(self) -> Dict[AnyStr, Any]:
        pass
    def generate_request_body(self):
        pass