import random
import string
from typing import Dict, List

from specification_parser import ParameterProperties, ItemProperties


class RandomizedSelector:
    def __init__(self, parameters: dict, request_body: dict):
        self.generate_accurate = random.randint(1, 10) <= 3
        self.dropout_ratio = 0.05
        self.randomized_weight = 0.8
        self.max_arr_length = 2**32
        self.randomization_max_val = 100
        self.generators = {"integer": self.randomize_integer,
                           "float": self.randomize_float,
                           "number": self.randomize_float,
                           "boolean": self.randomize_boolean,
                           "string" : self.randomize_string,
                           "array" : self.randomize_array,
                           "object" : self.randomize_object,
                           "null": self.randomize_null}
        self.parameters: Dict[str, ParameterProperties] = parameters
        self.request_body: Dict[str, ItemProperties] = request_body

    def use_primitive_generator(self, item_properties: ItemProperties):
        if item_properties is None:
            return None
        generator = self.generators.get(item_properties.type)
        if self.should_generate_accurately():
            return generator() if generator else None
        else:
            return random.choice(list(self.generators.values()))()

    def generate_randomized_object(self, item_properties: ItemProperties) -> Dict:
        if not item_properties.properties:
            return self.generators["object"]()
        randomized_object = {}
        for item_name, item_values in item_properties.properties.items():
            if not self.is_dropped():
                randomized_object[item_name] = self.randomize_item(item_values)
        return randomized_object

    def generate_randomized_array(self, item_properties: ItemProperties) -> List:
        if not item_properties.items:
            return self.generators["array"]()
        array_length = self.randomized_array_length()
        return [self.randomize_item(item_properties.items) for _ in range(array_length)]

    def randomize_item(self, item_properties: ItemProperties):
        if item_properties is None:
            return None
        if item_properties.type in ["object", "array"] and self.should_generate_accurately():
            composite_method = self.generate_randomized_object if item_properties.type == "object" else self.generate_randomized_array
            return composite_method(item_properties)
        else:
            return self.use_primitive_generator(item_properties)

    def randomize_parameters(self) -> Dict[str, any]:
        query_parameters = {}
        for parameter_name, parameter_properties in self.parameters.items():
            if self.is_dropped():
                continue
            randomized_value = self.randomize_item(parameter_properties.schema)
            query_parameters[parameter_name] = randomized_value
        return query_parameters

    def randomize_request_body(self):
        if self.is_dropped():
            return []
        if isinstance(self.request_body, list):
            return [self.randomize_item(item) for item in self.request_body]
        elif isinstance(self.request_body, ItemProperties):
            return self.randomize_item(self.request_body)
        else:
            error_msg = "Error parsing request body"
            print(error_msg)
            raise ValueError(error_msg)

    def should_generate_accurately(self):
        return self.generate_accurate or not self.randomize_type()

    def randomize_type(self):
        return random.randint(1, self.randomization_max_val) < self.randomized_weight * self.randomization_max_val # return accurate

    def is_dropped(self):
        return random.randint(0, self.randomization_max_val) < self.dropout_ratio * self.randomization_max_val if not self.generate_accurate else False
        
    def randomize_integer(self):
        return random.randint(-2**32, 2**32)

    def randomize_float(self):
        return random.uniform(-2**32, 2**32)

    def randomize_boolean(self):
        return random.choice([True, False])

    def randomize_string(self):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 9999)))

    def randomize_array(self):
        return [random.randint(-9999, 9999) for _ in range(random.randint(1, 9999))]

    def randomize_object(self):
        return {random.choice(string.ascii_letters): random.randint(-9999, 9999) for _ in range(random.randint(1, 9999))}

    def randomize_null(self):
        return None
    
    def randomized_array_length(self):
        array_size = random.randint(0, 100)
        if array_size <= 95:
            return random.randint(0, 1000)
        else:
            return random.randint(0, 2**32)
