import collections
import difflib
import json
import re

import numpy as np
from model.method import Method
from fuzzer.schema_validator import validate
from .util import resolve_json_value
from fuzzer.relation_reasoner import RelationReasoner


class ParameterType:
    STRING = "STRING"
    BOOLEAN = "BOOLEAN"
    NUMBER = "NUMBER"
    NONE = "NONE"
    OBJECT = "OBJECT"
    LIST = "LIST"


class RuntimeDictionary:
    def __init__(self):
        self.api_response_parameter = {}
        self.signature_to_value = {}
        self.signature_splitter = '%APITestGen%'
        self.array_pattern = re.compile('\[[0-9]+\]')
        self.path_to_signature_threshold = {}
        self.path_to_signature_suffix = {}
        self.path_to_type = {}
        self.signature_value_match_dict = {}
        self.debug_method = "getLunDetails"
        self.relation_reasoner = RelationReasoner()

    def remove_array_notation(self, key=""):
        notations = self.array_pattern.findall(key)
        for notation in notations:
            key = key.replace(notation, "[0]")
        return key

    def has_candidate_in_dictionary(self, path: str):
        return self.path_to_signature_threshold.__contains__(path) or self.path_to_signature_suffix.__contains__(path)

    def calculate_path_threshold(self, path: str):
        sequence_matcher = difflib.SequenceMatcher()
        for signature in self.signature_to_value.keys():
            api, response_value_path = signature.split(self.signature_splitter)
            # ratio
            sequence_matcher.set_seqs(response_value_path, path)
            signature_threshold_list = self.path_to_signature_threshold.get(path, [])
            signature_threshold_list.append((signature, sequence_matcher.ratio()))
            self.path_to_signature_threshold[path] = signature_threshold_list

            # suffix match
            ignored_response_value_path = str(response_value_path).lower().split('.')[-1]
            ignored_case_path = str(path).lower().split('.')[-1]
            if ignored_response_value_path.endswith(ignored_case_path) or ignored_case_path.endswith(
                    ignored_response_value_path):
                suffix_set = self.path_to_signature_suffix.get(path, set())
                suffix_set.add(signature)
                self.path_to_signature_suffix[path] = suffix_set

    def update_signature_to_path_threshold(self, signature):
        api, response_value_path = signature.split(self.signature_splitter)
        sequence_matcher = difflib.SequenceMatcher()
        for path in self.path_to_signature_threshold.keys():
            sequence_matcher.set_seqs(path, response_value_path)
            self.path_to_signature_threshold[path].append((signature, sequence_matcher.ratio()))
            ignored_case_response_path = str(response_value_path).lower().split('.')[-1]
            ignored_case_path = str(path).lower().split('.')[-1]
            # suffix match
            if ignored_case_path.endswith(ignored_case_response_path) or ignored_case_response_path.endswith(
                    ignored_case_path):
                suffix_set = self.path_to_signature_suffix.get(path, set())
                suffix_set.add(signature)
                self.path_to_signature_suffix[path] = suffix_set

    def should_use_dictionary(self, path: str):
        # we leave the opportunity for random
        if np.random.random() < 0.1:
            return False
        if self.path_to_signature_suffix.__contains__(path) or self.path_to_signature_threshold.__contains__(path):
            return True
        return False

    def generate_value_from_dictionary(self, path: str, method: Method, schema: dict):
        feed_from_methods = [feed_method.method_name for feed_method in method.feed_from_method]
        valid_parameters = []
        feed_from_valid_parameters = []
        odg_reference_signature = []
        if "existing_lun_ids[0]" in path:
            a = 2
        if "deleteLunGroup" == method.method_name:
            c = 3
        for method_name in method.required_feed_parameter:
            dependency_dict = method.required_feed_parameter[method_name]
            for reference in dependency_dict:
                reference_attribute_list = method.get_single_request_parameter_by_property_name(reference)
                feed_parameter_list = method.get_feed_from_method_response_parameter_by_property_and_method_name(
                    method_name, dependency_dict[reference])
                if path in reference_attribute_list and len(feed_parameter_list) > 0:
                    for feed_parameter in feed_parameter_list:
                        signature = f'{method_name}{self.signature_splitter}{feed_parameter}'
                        if signature in self.signature_to_value:
                            odg_reference_signature.append(signature)
        if len(odg_reference_signature) > 0 and np.random.random() > 0.05:
            valid_references = []
            for signature in odg_reference_signature:
                sample = self.signature_to_value[signature][
                    np.random.randint(0, len(self.signature_to_value[signature]))]
                try:
                    validate(sample, schema)
                    valid_references.append(signature)
                except:
                    pass
            if len(valid_references) > 0:
                signature = valid_references[np.random.randint(0, len(valid_references))]
                val = self.signature_to_value[signature][np.random.randint(0, len(self.signature_to_value[signature]))]
                return val
        for parameter_signature in self.signature_to_value:
            sample_value = self.signature_to_value[parameter_signature][0]
            try:
                validate(sample_value, schema)
                # if is_id and not ('id' in parameter_signature):
                #     continue

                # should_skip = True
                # for method_name in feed_from_methods:
                #     if method_name in parameter_signature:
                #         should_skip = True
                #         break
                # if should_skip:
                #     continue
                valid_parameters.append(parameter_signature)
                for feed_from_method_name in feed_from_methods:
                    if feed_from_method_name in parameter_signature:
                        feed_from_valid_parameters.append(parameter_signature)
                        break
            except Exception as ex:
                pass
        if len(valid_parameters) == 0:
            return None
        suffix_parameters = self.path_to_signature_suffix.get(path, set())
        intersection_suffix_feed_from_parameters = set.intersection(suffix_parameters, set(feed_from_valid_parameters))
        valid_suffix_parameters = set.intersection(suffix_parameters, feed_from_valid_parameters)
        # we leave the opportunity for other cases
        if len(intersection_suffix_feed_from_parameters) > 0 and np.random.random() < 0.5:
            signature = np.random.choice(list(intersection_suffix_feed_from_parameters))
            value = self.signature_to_value[signature][np.random.choice(len(self.signature_to_value[signature]))]
            return value
        # suffix parameters
        if len(valid_suffix_parameters) > 0 and np.random.random() < 0.5:
            signature = np.random.choice(list(valid_suffix_parameters))
            value = self.signature_to_value[signature][np.random.choice(len(self.signature_to_value[signature]))]
            return value

        # valid parameters
        if len(valid_parameters) > 0:
            signature = np.random.choice(list(valid_parameters))
            value = self.signature_to_value[signature][np.random.choice(len(self.signature_to_value[signature]))]
            return value

        # from threshold dictionary
        weights = []
        candidates = []
        for item in self.path_to_signature_threshold[path]:
            cand, w = item
            if not (cand in valid_parameters):
                continue
            weights.append(w + 0.00001)
            candidates.append(cand)
        weights = weights / np.sum(weights)
        signature = np.random.choice(candidates, 1, p=weights)[-1]
        value = self.signature_to_value[signature][np.random.choice(len(self.signature_to_value[signature]))]
        # print('in threshold', path, signature, value)
        return value

    def inject_signature(self, api_name, attribute_path, value):
        signature = api_name + self.signature_splitter + attribute_path
        fifo = self.signature_to_value.get(signature, collections.deque(maxlen=20))
        self.signature_to_value[signature] = fifo
        fifo.append(value)

    def infer_response_value_type(self, val):
        if isinstance(val, bool):
            return ParameterType.BOOLEAN
        elif isinstance(val, dict):
            return ParameterType.OBJECT
        elif isinstance(val, list):
            return ParameterType.LIST
        elif isinstance(val, float) or isinstance(val, int):
            return ParameterType.NUMBER
        elif isinstance(val, str):
            return ParameterType.STRING
        elif val is None:
            return ParameterType.NONE
        raise Exception('unknown value type {}'.format(val))

    def parse(self, method, response={}):
        has_new_attribute = False
        # we do not consider abnormal responses
        if response["statusCode"] > 299:
            return has_new_attribute

        content = response["content"]
        api_name = method.method_name

        # we do not handle empty content
        if content is None or len(content) == 0:
            return has_new_attribute

        try:
            content = json.loads(content)
            data = {}
            resolve_json_value("", content, data)
            # self.relation_reasoner.update_or_create_db_by_data(request_value, variable_replace_map, method, data)
            for key in data.keys():
                value = data[key]
                nominal_key = self.remove_array_notation(key)
                signature = f'{api_name}{self.signature_splitter}{nominal_key}'
                signature_to_value = self.signature_to_value.get(signature, collections.deque(maxlen=20))
                if not self.signature_to_value.__contains__(signature):
                    self.signature_to_value[signature] = signature_to_value
                    self.update_signature_to_path_threshold(signature)
                    has_new_attribute = True
                    print(signature, value)
                # record value type
                # value_type_set = self.path_to_type.get(signature, set())
                # value_type_set.add(value_type)
                # self.path_to_type[signature] = value_type_set
                # if value_type in [ParameterType.STRING]:
                #     for sig in self.path_to_type:
                #         if api_name in sig or not (value_type in self.path_to_type[sig]):
                #             continue
                #         compare_sig_set = self.signature_value_match_dict.get(sig, set())
                #         current_sig_set = self.signature_value_match_dict.get(signature, set())
                #         if sig in current_sig_set:
                #             continue
                #         for val in self.signature_to_value[sig]:
                #             if val == value:
                #                 compare_sig_set.add(signature)
                #                 current_sig_set.add(sig)
                #                 break
                #         self.signature_value_match_dict[sig] = compare_sig_set
                #         self.signature_value_match_dict[signature] = current_sig_set
                if not (value in signature_to_value):
                    signature_to_value.append(value)
            return has_new_attribute
        except Exception as e:
            print("exception", content)
            print(e)
            # raise e
            return has_new_attribute
