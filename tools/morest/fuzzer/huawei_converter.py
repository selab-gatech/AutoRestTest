import datetime
import json
import logging
import uuid

import numpy as np

from model.sequence import Sequence
from .request_builder import build_request
from .runtime_dictionary import RuntimeDictionary
from .test_data_generator import RandomDataGenerator

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class RequestBodyType:
    JSON = "1"
    TEXT = "2"


class ContentType:
    XFORM = "1"
    JSON = "2"


class VariableGetterType:
    ALL = 1
    KEY_VALUE = 2
    JSON = 3
    REG_EXPRESSION = 4
    TEXT = 5


class HuaWeiConverter:

    def __init__(self, runtime_dictionary: RuntimeDictionary):
        self.ref_prefix = "%{APIGen#"
        self.ref_suffix = "}"
        self.runtime_dictionary = runtime_dictionary

    def get_date_time(self):
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        return date_time

    def generate_random_parameter(self, index, method, sequence):
        values = []
        ref_dict = sequence.ref_vars.get(index, {})
        refs = set()
        for param_name in method.request_parameters:
            parameter = method.request_parameters[param_name]
            generator = RandomDataGenerator(parameter, self.runtime_dictionary, ref_dict)
            generated_value = generator.generate()
            refs = refs.union(generator.refs_varabile)
            values.append((parameter, generated_value))
        url, params, data, headers, files = build_request(method, values)
        return url, params, data, headers, files, refs

    def generate_api_variables(self, defs, method):
        results = []
        for param in defs:
            variable = {
                "expression": param,
                "getValueMethod": VariableGetterType.JSON,
                "hearderName": "",
                "matchNum": 0,
                "source": 1,
                "varName": f'{method.method_signature}{param}',
            }
            results.append(variable)
        return results

    def generate_test_step(self, fuzzer, sequence: [], method, url, params, data, headers, files, span_id, sequence_id,
                           request_id):
        kv_headers = []
        for key in headers.keys():
            kv_headers.append({
                "key": key,
                "value": headers[key]
            })
        result = {
            "apiID": method.method_id,
            "seedID": fuzzer.seed_id,
            "requestId": request_id,
            "sequenceID": sequence_id,
            "span_id": str(span_id),
            "apiName": method.method_name,
            "url": f'{fuzzer.server_address}{fuzzer.specification.get("basePath", "")}{url}',
            "method": f'{method.method_type.upper()}',
            "protocol": fuzzer.protocol,
            "timeLimit": fuzzer.time_limit,
            "rawBodyType": RequestBodyType.JSON,
            "requestBodyStr": json.dumps(data, cls=NpEncoder),
            "contentType": ContentType.JSON,
            "requestBodyMap": [
            ],
            "requestHeader": kv_headers,
            # "apiVariables": self.generate_api_variables(method)
        }
        if len(params.values()) > 0:
            result["url"] += "?" + "&".join([f'{key}={params[key]}' for key in params.keys()])
        return result

    def convert_sequence(self, fuzzer, sequence: Sequence):
        span_id = 0
        result = {
            "createdDate": self.get_date_time(),
            "iterationId": fuzzer.iteration_id,
            "serviceName": 'eoo',
        }
        scene_apis = []
        sequence_id = uuid.uuid4().__str__()
        result["sequenceID"] = sequence_id
        # print_path(sequence)
        # print("sequence id", sequence_id)

        for i, method in enumerate(sequence.requests):
            request_id = uuid.uuid4().__str__()
            url, params, data, headers, files, refs = self.generate_random_parameter(i, method,
                                                                                         sequence)


            test_step = self.generate_test_step(fuzzer, sequence, method, url, params, data, headers, files, span_id,
                                                sequence_id, request_id)
            span_id += 1
            if len(refs) > 0:
                api_variables = self.generate_api_variables(refs, sequence.requests[i - 1])
                scene_apis[i - 1]["apiVariables"] = api_variables
            scene_apis.append(test_step)
        result["sceneApis"] = scene_apis
        return result
