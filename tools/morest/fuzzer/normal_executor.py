import datetime
import json
import logging
import uuid

import numpy as np

from model.sequence import Sequence
from .request_builder import build_request
from .runtime_dictionary import RuntimeDictionary
from .normal_test_data_generator import RandomDataGenerator
from .util import resolve_json_value
from model.reference_definition import ReferenceDefinition
import requests
import re

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


class SequenceConverter:

    def __init__(self, runtime_dictionary: RuntimeDictionary):
        self.ref_prefix = "%{APIGen#"
        self.ref_suffix = "}"
        self.runtime_dictionary = runtime_dictionary

    def get_date_time(self):
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d %H:%M:%S")
        return date_time

    def generate_random_parameter(self, index, method, sequence, ref_values):
        values = []
        ref_dict = sequence.ref_vars.get(index, ReferenceDefinition([]))
        for param_name in method.request_parameters:
            parameter = method.request_parameters[param_name]
            generator = RandomDataGenerator(parameter, self.runtime_dictionary, ref_dict, ref_values, method)
            generated_value = generator.generate()
            values.append((parameter, generated_value))
        url, params, data, headers, files, form_data = build_request(method, values)
        return url, params, data, headers, files, form_data

    def extract_value(self, response):
        status_code = response["statusCode"]
        resolved_value = {}
        if status_code < 200 or status_code > 300:
            return resolved_value
        content = response["content"]
        if content is None or len(content) == 0:
            return resolved_value
        content = json.loads(content)
        resolve_json_value("", content, resolved_value)
        return resolved_value

    def check_header_validity(self, header):
        """Verifies that header value is a string which doesn't contain
        leading whitespace or return characters. This prevents unintended
        header injection.
        :param header: tuple, in the format (name, value).
        """
        name, value = header
        # Moved outside of function to avoid recompile every call
        _CLEAN_HEADER_REGEX_BYTE = re.compile(b'^\\S[^\\r\\n]*$|^$')
        _CLEAN_HEADER_REGEX_STR = re.compile(r'^\S[^\r\n]*$|^$')
        if isinstance(value, bytes):
            pat = _CLEAN_HEADER_REGEX_BYTE
        else:
            pat = _CLEAN_HEADER_REGEX_STR
        try:
            if not pat.match(value):
                raise Exception("Invalid return character or leading space in header: %s" % name)
        except TypeError:
            raise Exception("Value for header {%s: %s} must be of type str or "
                            "bytes, not %s" % (name, value, type(value)))

    def do_request(self, session, url, method, params, data, headers, files, form_data, pre_defined_headers):
        response = {
            "content": "",
            "apiName": method.method_name,
            "request": {
                "url": url, "method": method.method_type, "params": params, "data": data, "headers": headers,
                "files": files
            }
        }
        executed_method = getattr(session, method.method_type)

        try:
            for ind, key in enumerate(headers):
                self.check_header_validity((key, headers[key]))
        except Exception as ex:
            print(ex)
            headers = {}
        # add pre defined headers
        for k in pre_defined_headers.keys():
            headers[k] = pre_defined_headers[k]

        # do request
        try:
            raw_response = executed_method(url, params=params, data=form_data, json=data, headers=headers, files=files,
                                           allow_redirects=False, timeout=3)
        except requests.exceptions.ReadTimeout as err:
            print(err)
            response["statusCode"] = 524
            response["content"] = "Timeout Error"
            response["resolved_value"] = "None"
        except Exception as e:  # probably an encoding error
            return None
        else:
            response["statusCode"] = raw_response.status_code
            response["content"] = raw_response.text
            try:
                resolved_value = self.extract_value(response)
            except:  # returned value format not correct
                resolved_value = {}
            response["resolved_value"] = resolved_value
        return response

    def execute_request(self, fuzzer, session, method, url, params, data, headers, files, form_data,
                        pre_defined_headers):
        request_url = fuzzer.server_address + url
        result = self.do_request(session, request_url, method, params, data, headers, files, form_data,
                                 pre_defined_headers)
        return result

    def execute_sequence(self, fuzzer, sequence: Sequence):
        result = []
        session = requests.session()
        last_call_response = {}
        for i, method in enumerate(sequence.requests):
            url, params, data, headers, files, form_data = self.generate_random_parameter(i, method,
                                                                                          sequence, last_call_response)

            resp = self.execute_request(fuzzer, session, method, url, params, data, headers, files, form_data,
                                        fuzzer.pre_defined_headers)
            result.append(resp)
            last_call_response = resp["resolved_value"]
        return result
