import datetime
import random
import string

import numpy as np
import rstr
from fuzzer.schema_validator import validate
from fuzzer.runtime_dictionary import RuntimeDictionary
from fuzzer.constant import ValueSource
from model.parameter import TargetType, TargetStatus
from model.method import Method
import uuid


class RandomDataGenerator:
    SKIP_OPTIONAL = "SKIP_OPTIONAL"

    def __init__(self, parameter, runtime_dictionary: RuntimeDictionary, ref_dict: {}, ref_values: {}, method: Method):
        self.parameter = parameter
        self.parameter_name = parameter.name
        self.runtime_dictionary = runtime_dictionary
        self.ref_dict = ref_dict
        self.ref_prefix = "%{APIGen"
        self.ref_suffix = "}"
        self.threshold = 0.7
        self.validate_value = False
        self.refs_varabile = set()
        self.ref_values = ref_values
        self.value_source_count = {}
        self.potential_targets = set()
        self.value_source = {}
        self.method = method

    def generate(self):
        if self.should_skip_optional_parameter(self.parameter.raw_body):
            self.record_source(ValueSource.skip)
            # self.value_source[self.parameter_name] = (self.parameter_name, ValueSource.skip)
            return RandomDataGenerator.SKIP_OPTIONAL
        if self.is_simple_request_parameter(self.parameter.raw_body):
            val = self.value_factory(self.parameter_name, self.parameter.raw_body)
        else:
            val = self.value_factory("", self.parameter.raw_body)
        # validate generated value
        if self.validate_value:
            validate(val, self.parameter.raw_body)
        # assert val is not None, f'{self.parameter.name}'
        return val

    def is_simple_request_parameter(self, parameter_body):
        if parameter_body.__contains__("in") and parameter_body["in"] == "body":
            return False
        return True

    def get_value_source_stat(self):
        return self.value_source_count

    def record_source(self, source_type):
        self.value_source_count[source_type] = self.value_source_count.get(source_type, 0) + 1

    def _build_ref(self, prop):
        value = None
        matched = False
        for ref_path in prop:
            if ref_path in self.ref_values:
                value = self.ref_values[ref_path]
                matched = True
                return value, matched
        return value, matched

    def should_use_example(self, path, parameter_body):
        # only for debug should be modified
        if isinstance(parameter_body, dict) and (parameter_body.__contains__('example') or parameter_body.__contains__(
                'x-example')) and np.random.random() < 0.5:
            return True
        return False

    def should_skip_optional_parameter(self, parameter_body):
        if parameter_body.__contains__('required') and parameter_body['required'] is True:
            return False
        if parameter_body.__contains__('in') and parameter_body['in'] in ['path', 'body']:
            return False
        # only for debug should be modified
        if np.random.random() < 0.4:
            return False
        return True

    def should_skip_optional_property(self, parameter_body, key):
        if parameter_body.__contains__('required') and key in parameter_body['required']:
            return False
        if parameter_body.__contains__('in') and parameter_body['in'] in ['path', 'body']:
            return False
        if np.random.random() < 0.4:
            return False
        return True

    def use_example(self, path, parameter_body):
        potential_target = (path, TargetType.EXAMPLE, TargetStatus.EXAMPLE)
        self.potential_targets.add(potential_target)
        if parameter_body.__contains__('example'):
            example = parameter_body['example']
        else:
            example = parameter_body['x-example']
        val = example
        return val

    def value_factory(self, path, parameter):
        if self.should_use_example(path, parameter):
            val = self.use_example(path, parameter)
            self.record_source(ValueSource.example)
            self.value_source[path] = (val, ValueSource.example)
            return val
        if path in self.ref_dict:
            ref, matched = self._build_ref(self.ref_dict[path])
            if matched:
                return ref
        param_type = self.check_parameter_type(path, parameter)
        if param_type == "integer":
            val = self.integer_factory(path, parameter)
            assert val is not None, 'val(integer) is None'f'{path}'
            return val
        elif param_type == "string":
            val = self.string_factory(path, parameter)
            assert val is not None, 'val(string) is None'f'{path}'
            return val
        elif param_type == "file":
            val = self.string_factory(path, parameter)
            assert val is not None, 'val(file) is None'f'{path}'
            return val
        elif param_type == 'array':
            if self.should_use_example(path, parameter):
                val = self.use_example(path, parameter)
                self.record_source(ValueSource.example)
                self.value_source[path] = (val, ValueSource.example)
                return val
            if self.should_use_dictionary_value(path):
                value = self.runtime_dictionary.generate_value_from_dictionary(path, self.method, parameter)
                if isinstance(value, list):
                    try:
                        validate(value, parameter)
                        self.record_source(ValueSource.dictionary)
                        self.value_source[path] = (value, ValueSource.dictionary)
                        return value
                    except Exception:
                        pass
            items = parameter["items"]
            array = []
            for i in range(np.random.choice(range(0, 2))):
                item = self.value_factory(f'{path}[0]', items)
                if self.should_skip_optional_parameter(items):
                    path_signature = f'{path}[{i}]_skip'
                    self.record_source(ValueSource.skip)
                    # self.value_source[path_signature] = (item, ValueSource.skip)
                    continue
                assert item is not None, f'val(array,item) is None'f'{path}'
                array.append(item)
                self.value_source[f'{path}[0]'] = (item, ValueSource.random)
            # # fix for array's notation
            # if not path.endswith('[0]'):
            #     path += '[0]'
            if len(array) == 0:
                potential_target = (path, TargetType.ARRAY, TargetStatus.EMPTY)
            else:

                potential_target = (path, TargetType.ARRAY, TargetStatus.NON_EMPTY)
            self.potential_targets.add(potential_target)
            self.value_source[path] = (array, ValueSource.random)
            return array
        elif param_type == 'schema':
            val = self.build_schema(path, parameter["schema"])
            assert val is not None, f'val(array,item) is None'f'{path},{parameter}'
            return val
        elif param_type == "properties":
            parameter['type'] = 'object'
            val = self.build_schema(path, parameter)
            assert val is not None, f'val(properties) is None'f'{path}'
            return val
        elif param_type == 'object':
            val = self.object_factory(path, parameter)
            assert val is not None, f'val(object) is None'f'{path}'
            return val
        elif param_type == 'boolean':
            val = self.boolean_factory(path)
            assert val is not None, f'val(boolean) is None'f'{path}'
            return val
        elif param_type == "allOf":
            if self.should_use_example(path, parameter):
                val = self.use_example(path, parameter)
                self.record_source(ValueSource.example)
                self.value_source[path] = (val, ValueSource.example)
                return val
            if path in self.ref_dict:
                ref, matched = self._build_ref(self.ref_dict[path])
                if matched:
                    return ref
            if self.should_use_dictionary_value(path):
                value = self.runtime_dictionary.generate_value_from_dictionary(path, self.method, parameter)
                if isinstance(value, dict):
                    try:
                        validate(value, parameter)
                        self.record_source(ValueSource.dictionary)
                        self.value_source[path] = (value, ValueSource.dictionary)
                        return value
                    except:
                        pass
            result = {}
            for item in parameter["allOf"]:
                elem = self.value_factory(path, item)
                # if self.should_skip_optional_parameter(item):
                #     self.value_source[path] = (path, ValueSource.skip)
                #     continue
                assert elem is not None, f'val(allOf) is None'f'{path}'
                for key in elem.keys():
                    if self.should_skip_optional_property(item, key):
                        path_signature = f'{path}.{key}'
                        self.record_source(ValueSource.skip)
                        # self.value_source[path_signature] = (path, ValueSource.skip)
                        continue
                    result[key] = elem[key]
            self.value_source[path] = (result, ValueSource.random)
            return result
        elif param_type == 'number':
            val = self.number_factory(path, parameter)
            assert val is not None, 'val(number) is None'f'{path}'
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
        if self.should_use_example(path, schema):
            val = self.use_example(path, schema)
            self.record_source(ValueSource.example)
            return val
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
            if self.should_use_example(path, schema):
                self.record_source(ValueSource.example)
                val = self.use_example(path, schema)
                self.value_source[path] = (val, ValueSource.example)
                return val
            if path in self.ref_dict:
                ref, matched = self._build_ref(self.ref_dict[path])
                if matched:
                    return ref

            if self.should_use_dictionary_value(path):
                value = self.runtime_dictionary.generate_value_from_dictionary(path, self.method, schema)
                if isinstance(value, list):
                    try:
                        validate(value, schema)
                        self.record_source(ValueSource.dictionary)
                        self.value_source[path] = (value, ValueSource.dictionary)
                        return value
                    except:
                        pass
            items = schema["items"]
            array = []
            for i in range(np.random.choice(range(0, 2))):
                res = self.value_factory(f'{path}[0]', items)
                if self.should_skip_optional_parameter(items):
                    path_signature = f'{path}[0]'
                    self.record_source(ValueSource.skip)
                    # self.value_source[path_signature] = (path, ValueSource.skip)
                    continue
                array.append(res)
                self.value_source[f'{path}[0]'] = (res, ValueSource.random)

            if len(array) == 0:
                potential_target = (path, TargetType.ARRAY, TargetStatus.EMPTY)
            else:
                potential_target = (path, TargetType.ARRAY, TargetStatus.NON_EMPTY)
            self.potential_targets.add(potential_target)
            self.value_source[path] = (array, ValueSource.random)
            return array
        if schema_type == 'allOf':
            if self.should_use_example(path, schema):
                val = self.use_example(path, schema)
                self.record_source(ValueSource.example)
                self.value_source[path] = (val, ValueSource.example)
                return val
            if path in self.ref_dict:
                ref, matched = self._build_ref(self.ref_dict[path])
                if matched:
                    return ref
            if self.should_use_dictionary_value(path):
                value = self.runtime_dictionary.generate_value_from_dictionary(path, self.method, schema)
                if isinstance(value, dict):
                    try:
                        validate(value, schema)
                        self.record_source(ValueSource.dictionary)
                        self.value_source[path] = (value, ValueSource.dictionary)
                        return value
                    except:
                        pass
            result = {}
            for item in schema["allOf"]:
                elem = self.value_factory(path, item)
                # if self.should_skip_optional_parameter(item):
                #     path_signature = f'{path}'
                #     self.value_source[path_signature] = (path, ValueSource.skip)
                #     continue
                assert elem is not None, f'val(allOf) is None'f'{path}'
                for key in elem.keys():
                    if self.should_skip_optional_property(item, key):
                        path_signature = f'{path}.{key}'
                        self.record_source(ValueSource.skip)
                        # self.value_source[path_signature] = (path, ValueSource.skip)
                        continue
                    result[key] = elem[key]
            self.value_source[path] = (result, ValueSource.random)
            return result
        val = self.value_factory(path, schema)
        self.value_source[path] = (val, ValueSource.random)
        return val

    def should_use_dictionary_value(self, path: str):
        if not self.runtime_dictionary.has_candidate_in_dictionary(path):
            self.runtime_dictionary.calculate_path_threshold(path)
        # if len(self.runtime_dictionary.signature_to_value) > 0 and str.lower(path).endswith("id"):
        #     return True
        return self.runtime_dictionary.should_use_dictionary(path)

    def string_factory(self, path, string_body={}):
        if self.should_use_example(path, string_body):
            val = self.use_example(path, string_body)
            self.record_source(ValueSource.example)
            self.value_source[path] = (val, ValueSource.example)
            return val
        if path in self.ref_dict:
            ref, matched = self._build_ref(self.ref_dict[path])
            if matched:
                return ref
        min_len = 0
        max_len = 100
        self.record_source(ValueSource.random)
        if string_body.__contains__("enum"):
            self.record_source(ValueSource.enum)
            enum = np.random.choice(string_body["enum"])
            potential_target = (path, TargetType.ENUM, enum)
            self.potential_targets.add(potential_target)
            self.value_source[path] = (enum, ValueSource.enum)
            return enum

        if self.should_use_dictionary_value(path):
            value = self.runtime_dictionary.generate_value_from_dictionary(path, self.method, string_body)
            if isinstance(value, str):
                self.record_source(ValueSource.dictionary)
                self.value_source[path] = (value, ValueSource.dictionary)
                return value
        if string_body.__contains__("minLength") and string_body.__contains__("maxLength"):
            min_len = string_body["minLength"]
            max_len = string_body["maxLength"]

        if string_body.__contains__("minLength") and np.random.random() < 0.2:
            max_len = string_body["minLength"]
            potential_target = (path, TargetType.STRING, TargetStatus.MIN)
            self.potential_targets.add(potential_target)

        elif string_body.__contains__("maxLength") and np.random.random() < 0.2:
            min_len = string_body["maxLength"]
            potential_target = (path, TargetType.STRING, TargetStatus.MAX)
            self.potential_targets.add(potential_target)

        elif string_body.__contains__("minLength") and string_body.__contains__("maxLength"):
            potential_target = (path, TargetType.STRING, TargetStatus.MIDDLE)
            self.potential_targets.add(potential_target)

        if string_body.__contains__('format'):
            self.record_source(ValueSource.random)
            format = string_body['format']
            if format == 'date-time':
                res = datetime.datetime.now().isoformat('T')
                self.value_source[path] = (res, ValueSource.random)
                return res
            elif format == 'uuid':
                res = uuid.uuid4().__str__()
                self.value_source[path] = (res, ValueSource.random)
                return res
            elif format == "password":
                res = "testpassword"
                return res
            else:
                raise Exception('unknown string format', format)
        if string_body.__contains__('pattern'):
            self.record_source(ValueSource.random)
            pattern = string_body['pattern']
            res = rstr.xeger(pattern)
            self.value_source[path] = (res, ValueSource.random)
            return res
        # avoid body size too large
        max_len = min(max_len, 100)
        if max_len <= min_len:
            str_len = max_len
        else:
            str_len = np.random.randint(min_len, max_len + 1)
        res = ''.join(random.choices(string.ascii_uppercase +
                                     string.digits, k=str_len))
        if str_len == min_len:
            self.record_source(ValueSource.min)
            self.value_source[path] = (res, ValueSource.min)
        elif str_len == max_len:
            self.record_source(ValueSource.max)
            self.value_source[path] = (res, ValueSource.max)
        else:
            self.record_source(ValueSource.middle)
            self.value_source[path] = (res, ValueSource.middle)
        return res

    def object_factory(self, path, object_body={}):
        if self.should_use_example(path, object_body):
            val = self.use_example(path, object_body)
            self.record_source(ValueSource.example)
            self.value_source[path] = (val, ValueSource.example)
            return val
        if path in self.ref_dict:
            ref, matched = self._build_ref(self.ref_dict[path])
            if matched:
                return ref
        if self.should_use_dictionary_value(path):
            value = self.runtime_dictionary.generate_value_from_dictionary(path, self.method, object_body)
            if isinstance(value, dict):
                try:
                    validate(value, object_body)
                    self.record_source(ValueSource.dictionary)
                    self.value_source[path] = (value, ValueSource.dictionary)
                    return value
                except:
                    pass
        self.record_source(ValueSource.random)
        data = {}
        if object_body.__contains__("properties"):
            for name in object_body["properties"]:
                if len(path) > 0:
                    property_path = f'{path}.{name}'
                else:
                    property_path = name
                # add array notation
                # if self.check_parameter_type(path, object_body["properties"][name]) == 'array':
                #     property_path += '[0]'
                res = self.value_factory(property_path, object_body["properties"][name])
                if self.should_skip_optional_property(object_body,
                                                      name):
                    if len(path) > 0:
                        path_signature = f'{path}.{name}'
                    else:
                        path_signature = name
                    self.record_source(ValueSource.skip)
                    # self.value_source[path_signature] = (path, ValueSource.skip)
                    continue
                data[name] = res
        else:
            for name in object_body.keys():
                if name == 'additionalProperties':
                    continue
                if len(path) > 0:
                    property_path = f'{path}.{name}'
                else:
                    property_path = name
                # add array notation
                # if self.check_parameter_type(path, object_body[name]) == 'array':
                #     property_path += '[0]'
                # we omit additional properties generation
                # data[name] = self.value_factory(property_path, object_body[name])
                # # FIXME: should be properly handled
                # if str(name).lower().__contains__("time"):
                #     now = datetime.datetime.now()
                #     date_time = now.strftime("%H:%M:%S")
                #     data[name] = date_time
                # if str(name).lower().__contains__("date") and np.random.random() < 0.5:
                #     data[name] = 0
                # if str(name).lower().__contains__("policy_type") and np.random.random() < 0.5:
                #     data[name] = 1
                # if str(name).lower().__contains__("bw") and np.random.random() < 0.5:
                #     data[name] = np.random.randint(0, 999999)
        self.value_source[path] = (data, ValueSource.random)
        return data

    def boolean_factory(self, path):
        if path in self.ref_dict:
            ref, matched = self._build_ref(self.ref_dict[path])
            if matched:
                return ref
        res = np.random.choice(['true', 'false'])
        if res == 'true':
            potential_target = (path, TargetType.BOOL, TargetStatus.TRUE)
        else:
            potential_target = (path, TargetType.BOOL, TargetStatus.FALSE)
        self.potential_targets.add(potential_target)
        self.value_source[path] = (res, ValueSource.random)
        return res

    def number_factory(self, path, number_body={}):
        if self.should_use_example(path, number_body):
            val = self.use_example(path, number_body)
            self.record_source(ValueSource.example)
            self.value_source[path] = (val, ValueSource.example)
            return val
        if path in self.ref_dict:
            ref, matched = self._build_ref(self.ref_dict[path])
            if matched:
                return ref

        if number_body.__contains__("enum"):
            self.record_source(ValueSource.enum)
            enum = np.random.choice(number_body["enum"])
            potential_target = (path, TargetType.ENUM, enum)
            self.potential_targets.add(potential_target)
            self.value_source[path] = (enum, ValueSource.enum)
            return enum

        if self.should_use_dictionary_value(path):
            value = self.runtime_dictionary.generate_value_from_dictionary(path, self.method, number_body)
            if isinstance(value, float):
                self.record_source(ValueSource.dictionary)
                self.value_source[path] = (value, ValueSource.dictionary)
                return value
        # bypass for enum
        if np.random.random() < 0.5:
            res = np.random.randint(0, 2)
            self.value_source[path] = (res, ValueSource.random)
            return res

        if number_body.__contains__("minimum") and number_body.__contains__("maximum"):
            if np.random.random() < 0.5:
                res = np.random.randint(number_body["minimum"], number_body["maximum"], dtype=np.int64)
                potential_target = (path, TargetType.NUM, TargetStatus.MIDDLE)
                self.potential_targets.add(potential_target)
                self.value_source[path] = (res, ValueSource.middle)
                self.record_source(ValueSource.middle)
            else:
                res = np.random.choice([number_body["minimum"], number_body["maximum"]])
                if res == number_body['minimum']:
                    potential_target = (path, TargetType.NUM, TargetStatus.MIN)
                    self.record_source(ValueSource.min)
                    self.value_source[path] = (res, ValueSource.min)
                    self.value_source[path] = (res, ValueSource.min)
                else:
                    potential_target = (path, TargetType.NUM, TargetStatus.MAX)
                    self.record_source(ValueSource.max)
                    self.value_source[path] = (res, ValueSource.max)
                    self.potential_targets.add(potential_target)
                    self.value_source[path] = (res, ValueSource.max)
            return res
        elif number_body.__contains__("minimum"):
            if np.random.random() < 0.2:
                res = number_body["minimum"]
                potential_target = (path, TargetType.NUM, TargetStatus.MIN)
                self.potential_targets.add(potential_target)
                self.value_source[path] = (res, ValueSource.min)
                self.record_source(ValueSource.min)
            else:
                res = np.random.randint(0, 999999)
                self.value_source[path] = (res, ValueSource.middle)
                self.record_source(ValueSource.middle)
            return res
        elif number_body.__contains__("maximum"):
            if np.random.random() < 0.2:
                res = number_body["maximum"]
                potential_target = (path, TargetType.NUM, TargetStatus.MAX)
                self.potential_targets.add(potential_target)
                self.value_source[path] = (res, ValueSource.max)
                self.record_source(ValueSource.max)
            else:
                res = np.random.randint(0, 999999)
                self.value_source[path] = (res, ValueSource.middle)
                self.record_source(ValueSource.middle)
            return res
        else:
            res = np.random.randint(0, 999999)
            self.value_source[path] = (res, ValueSource.middle)
            self.record_source(ValueSource.middle)
            return res

    def integer_factory(self, path, integer_body={}):
        if self.should_use_example(path, integer_body):
            val = self.use_example(path, integer_body)
            self.record_source(ValueSource.example)
            self.value_source[path] = (val, ValueSource.example)
            return val
        if path in self.ref_dict:
            ref, matched = self._build_ref(self.ref_dict[path])
            if matched:
                return ref

        if integer_body.__contains__("enum"):
            self.record_source(ValueSource.enum)
            enum = np.random.choice(integer_body["enum"])
            potential_target = (path, TargetType.ENUM, enum)
            self.potential_targets.add(potential_target)
            self.value_source[path] = (enum, ValueSource.enum)
            return enum

        if self.should_use_dictionary_value(path):
            value = self.runtime_dictionary.generate_value_from_dictionary(path, self.method, integer_body)
            if isinstance(value, int):
                self.record_source(ValueSource.reference)
                self.value_source[path] = (value, ValueSource.reference)
                return value
        # bypass for enum
        if np.random.random() < 0.5:
            res = np.random.randint(0, 2)
            self.value_source[path] = (res, ValueSource.random)
            self.record_source(ValueSource.random)
            return res
        if integer_body.__contains__("minimum") and integer_body.__contains__("maximum"):
            if np.random.random() < 0.5:
                res = np.random.randint(integer_body["minimum"], integer_body["maximum"], dtype=np.int64)
                potential_target = (path, TargetType.NUM, TargetStatus.MIDDLE)
                self.potential_targets.add(potential_target)
                self.value_source[path] = (res, ValueSource.middle)
                self.record_source(ValueSource.middle)
            else:
                res = np.random.choice([integer_body["minimum"], integer_body["maximum"]])
                if res == integer_body['minimum']:
                    potential_target = (path, TargetType.NUM, TargetStatus.MIN)
                    self.value_source[path] = (res, ValueSource.min)
                    self.record_source(ValueSource.min)
                else:
                    potential_target = (path, TargetType.NUM, TargetStatus.MAX)
                    self.value_source[path] = (res, ValueSource.max)
                    self.record_source(ValueSource.max)
                self.potential_targets.add(potential_target)
            self.value_source[path] = (res, ValueSource.random)
            return res
        elif integer_body.__contains__("minimum"):
            if np.random.random() < 0.2:
                res = integer_body["minimum"]
                potential_target = (path, TargetType.NUM, TargetStatus.MIN)
                self.potential_targets.add(potential_target)
                self.value_source[path] = (res, ValueSource.min)
                self.record_source(ValueSource.min)
            else:
                res = np.random.randint(0, 999999)
                self.value_source[path] = (res, ValueSource.middle)
                self.record_source(ValueSource.middle)
            self.value_source[path] = (res, ValueSource.random)
            return res
        elif integer_body.__contains__("maximum"):
            if np.random.random() < 0.2:
                res = integer_body["maximum"]
                potential_target = (path, TargetType.NUM, TargetStatus.MAX)
                self.potential_targets.add(potential_target)
                self.value_source[path] = (res, ValueSource.max)
                self.record_source(ValueSource.max)
            else:
                res = np.random.randint(0, 999999)
                self.value_source[path] = (res, ValueSource.middle)
                self.record_source(ValueSource.middle)
            return res
        else:
            res = np.random.randint(0, 999999)
            self.value_source[path] = (res, ValueSource.random)
            self.record_source(ValueSource.random)
            return res
