import random
import string

import json 
import urllib 
from specification_parser import ItemProperties

class RandomizedSelector:
    def __init__(self):
        self.generate_accurate = random.randint(0, 10) < 2
        self.dropout_ratio = 0.05
        self.randomized_weight = 0.8
        self.max_arr_length = 2**32
        self.randomization_max_val = 100
        self.generators = {"int": self.randomize_integer,
                      "float" : self.randomize_float,
                      "bool": self.randomize_boolean,
                      "string" : self.randomize_string,
                      "array" : self.randomize_array,
                      "null": self.randomize_null}
        
    def generate_parameter_value(self, parameter_type):  
        if self.generate_accurate or random.randint(0, self.randomization_max_val) >= self.randomized_weight * self.randomization_max_val:
            return self.generators[parameter_type]()               
        else:
            return random.choice(list(self.generators.values()))()

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
        array_size = random.randint(1, 100)
        if array_size <= 95:
            return random.randint(0, 1000)
        else:
            return random.randint(1000, 2**32)
    
    def randomize_request_body(self, request_body):
        
        def convert_properties(self, object: ItemProperties):
            if object.type == "array" and not self.is_dropped():
                num_objects = self.randomized_array_length()
                obj_arr = []
                for _ in range(num_objects):
                    item = self.convert_properties(object.items)
                    if item:
                        obj_arr.append(item)
                return obj_arr
            elif object.type == "object" and not self.is_dropped():
                object_structure = {}
                for key, value in object.properties.items():
                    if not self.is_dropped():
                        object_structure[key] = self.convert_properties(value)
                    else:
                        continue
                return object_structure
            else:
                return self.generate_parameter_value(object.type)
        
        def convert_request_body(parsed_request_body):
            if 'application/json' in parsed_request_body:
                object = parsed_request_body['application/json']
                if isinstance(object, ItemProperties):
                    constructed_body = convert_properties(object)
                    return json.dumps(constructed_body)
                elif isinstance(object, list):
                    arr = []
                    for obj in object:
                        arr.append(convert_properties(obj))
                    return json.dumps(arr)
                else:
                    raise SyntaxError("Request Body Schema Parsing Error")
            elif 'application/x-www-form-urlencoded' in parsed_request_body:
                object = parsed_request_body['application/x-www-form-urlencoded']
                if isinstance(object, ItemProperties):
                    constructed_body = convert_properties(object)
                    return urllib.urlencode(constructed_body)
                elif isinstance(object, list):
                    arr = []
                    for obj in object:
                        arr.append(convert_properties(obj))
                    return urllib.urlencode(arr)
                else:
                    raise SyntaxError("Request Body Schema Parsing Error")
            else:
                keys = list(parsed_request_body.keys())
                if len(keys) == 1:
                    raise ValueError("Unsupported MIME type: " + keys[0] + " in Request Body Specification")
                else:
                    raise SyntaxError("Formatting Error in Specification")

        return convert_request_body(request_body)