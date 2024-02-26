import datetime
import random
import string
from utils.basic_payloads import general_payload_list

import numpy as np
import rstr

from fuzzer.runtime_dictionary import RuntimeDictionary


class PayloadDataGenerator:
    def __init__(self, parameter, runtime_dictionary: RuntimeDictionary = RuntimeDictionary(), ref_dict: {} = {},
                 ref_values: {} = {}):
        self.parameter = parameter
        self.parameter_name = parameter.name
        self.runtime_dictionary = runtime_dictionary
        self.ref_dict = ref_dict
        self.ref_prefix = "%{APIGen"
        self.ref_suffix = "}"
        self.threshold = 0.7
        self.refs_varabile = set()
        self.ref_values = ref_values
        self.payload_list = general_payload_list
        self.payload_list_length = len(general_payload_list)

    def generate(self):
        try:
            val = self.value_factory(self.parameter_name, self.parameter.raw_body)
        except Exception: # need to handle the errors more properly.
            val = None
        return val

    def can_use_example(self, body):
        return body.__contains__('example') and np.random.random() < 0.7

    def _build_ref(self, property):
        value = None
        matched = False
        for ref_path in property:
            if ref_path in self.ref_values:
                value = self.ref_values[ref_path]
                matched = True
                return value, matched
        return value, matched

    def _modify_example(self, example_value, param_type):
        # FIXME: Update the algorithm
        injection_val = None
        if param_type == "integer":
            injection_val = str(example_value) + random.choice(general_payload_list)

        elif param_type == "string":
            injection_val = random.choice(general_payload_list) # remove the example value for better injection

        elif param_type == "file":
            pass # do further processing

        elif param_type == 'array':
            print("Current param type is array; array content: ")
            print(example_value)
            raise Exception("Update code to handle the case.")

        elif param_type == 'schema':
            print("Current param type is schema; schema content: ")
            print(example_value)
            raise Exception("Update code to handle the case.")

        elif param_type == "properties":
            print("Current param type is schema; properties content: ")
            print(example_value)
            raise Exception("Update code to handle the case.")
        elif param_type == 'object':
            print("Current param type is schema; object content: ")
            print(example_value)
            raise Exception("Update code to handle the case.")

        elif param_type == 'boolean':
            pass

        elif param_type == "allOf":
            pass

        elif param_type == 'number':
            injection_val = str(example_value) + random.choice(general_payload_list)

        elif param_type == 'skip':
            pass

        else:
            raise Exception("Unrecognized type in parameter", param_type)
        
        if injection_val == None:
            return example_value
        else:
            return injection_val

    def value_factory(self, path, parameter, type_mix = False):
        if path in self.ref_dict:
            ref, matched = self._build_ref(self.ref_dict[path])
            if matched:
                return ref

        param_type = self.check_parameter_type(path, parameter)
        if self.can_use_example(parameter):
            example_value = parameter["example"] # cancel the original
            # return parameter["example"]
            modified_example = self._modify_example(example_value, param_type)
            return modified_example


        if param_type == "integer":
            # print("valpath: ", path, parameter)
            val = self.integer_factory(path, parameter, type_mix)
            # print("val: ", val)
            return val
        elif param_type == "string":
            val = self.string_factory(path, parameter)
            # print(val)
            return val
        elif param_type == "file":
            val = self.string_factory(path, parameter)
            return val
        elif param_type == 'array':
            if self.should_use_dictionary_value(path):
                value = self.runtime_dictionary.generate_value_from_dictionary(path)
                count = 0
                while (not isinstance(value, list)) and count < 10:
                    count += 1
                    value = self.runtime_dictionary.generate_value_from_dictionary(path)
                if isinstance(value, list):
                    return value
            if self.can_use_example(parameter):
                return parameter["example"]
            items = parameter["items"]
            array = []
            for i in range(np.random.choice(range(1, 3))):
                item = self.value_factory(f'{path}[0]', items)
                array.append(item)
            return array
        elif param_type == 'schema':
            val = self.build_schema(path, parameter["schema"])
            return val
        elif param_type == "properties":
            val = self.build_schema(path, parameter['properties'])
            return val
        elif param_type == 'object':
            val = self.object_factory(path, parameter)
            return val
        elif param_type == 'boolean':
            val = self.boolean_factory(path)
            return val
        elif param_type == "allOf":
            if path in self.ref_dict:
                ref, matched = self._build_ref(self.ref_dict[path])
                if matched:
                    return ref
            if self.can_use_example(parameter):
                return parameter["example"]
            if self.should_use_dictionary_value(path):
                value = self.runtime_dictionary.generate_value_from_dictionary(path)
                count = 0
                while (not isinstance(value, dict)) and count < 10:
                    count += 1
                    value = self.runtime_dictionary.generate_value_from_dictionary(path)
                if isinstance(value, dict):
                    return value
            result = {}
            for item in parameter["allOf"]:
                elem = self.value_factory(path, item)
                for key in elem.keys():
                    result[key] = elem[key]
            return result
        elif param_type == 'number':
            val = self.number_factory(path, parameter)
            return val
        elif param_type == 'skip':
            pass
        else:
            raise Exception("Unrecognized type in parameter", parameter)

    def check_parameter_type(self, path, parameter={}):
        param_type = None
        if parameter.__contains__("type") and isinstance(parameter['type'], str
                                                         ):
            param_type = parameter["type"]
        elif parameter.__contains__("schema"):
            param_type = "schema"
        elif parameter.__contains__("properties"):
            param_type = "properties"
        elif parameter.__contains__("allOf"):
            param_type = "allOf"
        elif parameter.__contains__('number'):
            param_type = 'number'
        # elif path.endswith('required'):
        #     param_type = 'skip'
        else:
            raise Exception(f"unknown parameter type: {parameter}")
        return param_type

    def build_schema(self, path, schema={}):
        if path in self.ref_dict:
            ref, matched = self._build_ref(self.ref_dict[path])
            if matched:
                return ref
        try:
            if isinstance(schema['type'], str):
                schema_type = schema["type"]
            else:
                raise Exception('not a schema type')
        except:
            if schema.__contains__("properties"):
                schema_type = "object"
            elif schema.__contains__("items"):
                schema_type = "array"
            elif schema.__contains__('allOf'):
                schema_type = 'allOf'
            else:
                schema_type = "object"
        if schema_type == "object":
            return self.object_factory(path, schema)
        if schema_type == "array":
            if path in self.ref_dict:
                ref, matched = self._build_ref(self.ref_dict[path])
                if matched:
                    return ref
            if self.can_use_example(schema):
                return schema["example"]
            if self.should_use_dictionary_value(path):
                value = self.runtime_dictionary.generate_value_from_dictionary(path)
                count = 0
                while (not isinstance(value, list)) and count < 10:
                    count += 1
                    value = self.runtime_dictionary.generate_value_from_dictionary(path)
                if isinstance(value, list):
                    return value
            items = schema["items"]
            array = []
            for i in range(np.random.choice(range(1, 3))):
                array.append(self.value_factory(f'{path}[0]', items))
            return array
        elif schema_type == 'allOf':
            if path in self.ref_dict:
                ref, matched = self._build_ref(self.ref_dict[path])
                if matched:
                    return ref
            if self.can_use_example(schema):
                return schema["example"]
            if self.should_use_dictionary_value(path):
                value = self.runtime_dictionary.generate_value_from_dictionary(path)
                count = 0
                while (not isinstance(value, dict)) and count < 10:
                    count += 1
                    value = self.runtime_dictionary.generate_value_from_dictionary(path)
                if isinstance(value, dict):
                    return value
            result = {}
            for item in schema["allOf"]:
                elem = self.value_factory(path, item)
                for key in elem.keys():
                    result[key] = elem[key]
            return result
        else:
            return self.value_factory(path, schema)

    def should_use_dictionary_value(self, path: str):
        if not self.runtime_dictionary.has_candidate_in_dictionary(path):
            self.runtime_dictionary.calculate_path_threshold(path)
        return self.runtime_dictionary.should_use_dictionary(path)

    def string_factory(self, path, string_body={}):
        if path in self.ref_dict:
            ref, matched = self._build_ref(self.ref_dict[path])
            if matched:
                return ref
        if np.random.random() > 0.6:
            # generate prefix integer first
            res = ''.join(random.choices(string.ascii_uppercase +
                                         string.digits, k=5))
            return res + random.choice(general_payload_list)
        elif np.random.random() < 0.1:
            # touch the special case to replace the result with the payload only
            return random.choice(general_payload_list)


        if self.can_use_example(string_body):
            return string_body["example"]
        min_len = 0
        max_len = 100
        if string_body.__contains__("enum"):
            return np.random.choice(string_body["enum"])
        if self.should_use_dictionary_value(path):
            value = self.runtime_dictionary.generate_value_from_dictionary(path)
            count = 0
            while (not isinstance(value, str)) and count < 10:
                count += 1
                value = self.runtime_dictionary.generate_value_from_dictionary(path)
            if isinstance(value, str):
                return value
        if string_body.__contains__("minLength") and string_body.__contains__("maxLength"):
            min_len = string_body["minLength"]
            max_len = string_body["maxLength"]
        if string_body.__contains__('format'):
            format = string_body['format']
            if format == 'date-time':
                res = datetime.datetime.now().isoformat('T')
                return res
            elif format == 'URL':
                return np.random.choice(['https://pypi.org/search/?q=random+url', 'https://google.com'])
            else:
                raise Exception('unknown string format ' + format)
        if string_body.__contains__('pattern'):
            pattern = string_body['pattern']
            res = rstr.xeger(pattern)
            return res
        str_len = np.random.randint(min_len, max_len + 1)
        res = ''.join(random.choices(string.ascii_uppercase +
                                     string.digits, k=str_len))
        return res

    def object_factory(self, path, object_body={}):
        if path in self.ref_dict:
            ref, matched = self._build_ref(self.ref_dict[path])
            if matched:
                return ref
        print(object_body)
        if self.can_use_example(object_body):
            return object_body["example"]
        if self.should_use_dictionary_value(path):
            value = self.runtime_dictionary.generate_value_from_dictionary(path)
            count = 0
            while (not isinstance(value, dict)) and count < 10:
                count += 1
                value = self.runtime_dictionary.generate_value_from_dictionary(path)
            if isinstance(value, dict):
                return value
        data = {}
        if object_body.__contains__("properties"):
            for name in object_body["properties"]:
                property_path = f'{path}.{name}'
                # add array notation
                # if self.check_parameter_type(path, object_body["properties"][name]) == 'array':
                #     property_path += '[0]'
                data[name] = self.value_factory(property_path, object_body["properties"][name])
        elif object_body.__contains__('type') and object_body['type'] == 'object' and not object_body.__contains__(
                'properties'):
            return {
                'key': self.string_factory("", {})
            }
        else:
            for name in object_body.keys():
                property_path = f'{path}.{name}'
                data[name] = self.value_factory(property_path, object_body[name])

        return data

    def boolean_factory(self, path):
        if path in self.ref_dict:
            ref, matched = self._build_ref(self.ref_dict[path])
            if matched:
                return ref
        return np.random.choice(['true', 'false'])

    def number_factory(self, path, number_body={}):
        if path in self.ref_dict:
            ref, matched = self._build_ref(self.ref_dict[path])
            if matched:
                return ref
        if self.can_use_example(number_body):
            return number_body["example"]
        if number_body.__contains__("enum"):
            return np.random.choice(number_body["enum"])
        if self.should_use_dictionary_value(path):
            value = self.runtime_dictionary.generate_value_from_dictionary(path)
            count = 0
            while isinstance(value, float) or isinstance(value, int) and count < 10:
                count += 1
                value = self.runtime_dictionary.generate_value_from_dictionary(path)
            return value
        if number_body.__contains__("minimum") and number_body.__contains__("maximum"):
            if np.random.random() < 0.5:
                return np.random.uniform(number_body["minimum"], number_body["maximum"])
            else:
                return np.random.choice([number_body["minimum"], number_body["maximum"]])
        else:
            return np.random.uniform(0, 999999)

    def integer_factory(self, path, integer_body={}, type_mix = False):
        if path in self.ref_dict:
            ref, matched = self._build_ref(self.ref_dict[path])
            if matched:
                return ref
        # in this mode the values should mostly from payload list (if not fixed)
        if np.random.random() > 0.2:  # test
            if type_mix == True:
                # generate prefix integer first
                if integer_body.__contains__("minimum") and integer_body.__contains__("maximum"):
                    if np.random.random() < 0.5:
                        prefix_value = np.random.randint(integer_body["minimum"], integer_body["maximum"], dtype=np.int64)
                    else:
                        prefix_value = np.random.choice([integer_body["minimum"], integer_body["maximum"]])
                else:
                    prefix_value = np.random.randint(0, 999999)
                return str(prefix_value) + random.choice(general_payload_list)

        if self.can_use_example(integer_body):
            return integer_body["example"]
        if integer_body.__contains__("enum"):
            return np.random.choice(integer_body["enum"])

        if self.should_use_dictionary_value(path):
            value = self.runtime_dictionary.generate_value_from_dictionary(path)
            count = 0
            while not isinstance(value, int) and count < 10:
                count += 1
                value = self.runtime_dictionary.generate_value_from_dictionary(path)
            return value

        if integer_body.__contains__("minimum") and integer_body.__contains__("maximum"):
            if np.random.random() < 0.5:
                return np.random.randint(integer_body["minimum"], integer_body["maximum"], dtype=np.int64)
            else:
                return np.random.choice([integer_body["minimum"], integer_body["maximum"]])
        else:
            return np.random.randint(0, 999999)
