import logging
import os
from typing import List, Dict, Tuple

from dataclasses import dataclass

import numpy as np
from gensim.downloader import load

from scipy.spatial.distance import cosine

from src.graph.specification_parser import OperationProperties, SchemaProperties

@dataclass
class SimilarityValue:
    dependent_val: str = ""
    in_value: str = ""
    similarity: float = 0.0

    def to_dict(self):
        return {
            "response": self.dependent_val,
            "in_value": self.in_value,
            "similarity": self.similarity
        }

class OperationDependencyComparator:
    def __init__(self):
        self.model = load("glove-wiki-gigaword-50")
        self.threshold = 0.7

    @staticmethod
    def handle_parameter_cases(parameter):
        reconstructed_parameter = []
        for index, char in enumerate(parameter):
            if char.isalpha():
                if char.isupper() and index != 0:
                    reconstructed_parameter.append(" " + char.lower())
                elif char == "_" or char == "-":
                    reconstructed_parameter.append(" ")
                else:
                    reconstructed_parameter.append(char)
        return "".join(reconstructed_parameter)

    @staticmethod
    def get_parameter_list(operation: OperationProperties) -> List[Dict[str,str]]:
        if operation.parameters is None:
            return []
        return [{OperationDependencyComparator.handle_parameter_cases(parameter): parameter}
                for parameter, parameter_details in operation.parameters.items()]

    @staticmethod
    def handle_schema_parameters(schema: SchemaProperties) -> List[Dict[str,str]]:
        object_params = []
        if schema.properties:
            for item, item_details in schema.properties.items():
                # Do check for "container" objects that are just an array of values
                if len(schema.properties) == 1 and item_details.type == "array":
                    object_params = OperationDependencyComparator.handle_schema_parameters(item_details.items)
                    
                else:
                    object_params.append({OperationDependencyComparator.handle_parameter_cases(item): item})
            return object_params
        elif schema.items:
            return OperationDependencyComparator.handle_schema_parameters(schema.items)
        return []

    @staticmethod
    def get_request_body_list(operation: OperationProperties) -> List[Dict[str,str]]:
        if operation.request_body is None:
            return []
        request_body_list = []
        for request_body_type, request_body_properties in operation.request_body.items():
            request_body_list += OperationDependencyComparator.handle_schema_parameters(request_body_properties)
        return request_body_list

    @staticmethod
    def get_response_list(operation: OperationProperties) -> List[Dict[str,str]]:
        if operation.responses is None:
            return []
        response_list = []
        for response_type, response_properties in operation.responses.items():
            if response_properties.content:
                for response, response_details in response_properties.content.items():
                    response_list += OperationDependencyComparator.handle_schema_parameters(response_details)
        return response_list

    def encode_sentence_or_word(self, thing: str):
        words = thing.split(" ")
        word_vectors = [self.model[word] for word in words if word in self.model]
        return np.mean(word_vectors, axis=0) if word_vectors else None

    def cosine_similarity(self, operation1_vals: List[Dict[str,str]], operation2_vals: List[Dict[str,str]], in_value: str = None) -> Dict[str, SimilarityValue]:
        """
        Returns parameters that might map between operations
        """
        parameter_response_similarity = {}
        for parameter_pairing in operation1_vals:
            for processed_parameter, parameter in parameter_pairing.items():
                for dependency_pairing in operation2_vals:
                    for processed_dependency, dependency in dependency_pairing.items():
                        parameter_encoding = self.encode_sentence_or_word(processed_parameter)
                        dependency_encoding = self.encode_sentence_or_word(processed_dependency)
                        if parameter_encoding is not None and dependency_encoding is not None:
                            similarity = 1 - cosine(parameter_encoding, dependency_encoding)
                            parameter_response_similarity[parameter] = SimilarityValue(
                                dependent_val=dependency,
                                in_value=in_value,
                                similarity=similarity
                            )

        return parameter_response_similarity

    def compare_response(self, operation1: OperationProperties, operation2: OperationProperties) -> (Dict[str, SimilarityValue], List[Tuple[str, SimilarityValue]]):
        parameter_matchings: Dict[str, SimilarityValue] = {}
        similar_parameters: Dict[str, SimilarityValue] = {}
        next_most_similar_parameters: List[(str, SimilarityValue)] = []

        operation1_parameters = self.get_parameter_list(operation1)
        operation1_body = self.get_request_body_list(operation1)
        operation2_parameters = self.get_parameter_list(operation2)
        operation2_body = self.get_request_body_list(operation2)
        operation2_responses = self.get_response_list(operation2)

        if operation1.parameters:
            if operation2.parameters:
                parameter_matchings = self.cosine_similarity(operation1_parameters, operation2_parameters, in_value="query to query")
            if operation2.request_body:
                parameter_matchings.update(self.cosine_similarity(operation1_parameters, operation2_body, in_value="query to body"))
            if operation2.responses:
                parameter_matchings.update(self.cosine_similarity(operation1_parameters, operation2_responses, in_value="query to response"))

        if operation1.request_body:
            if operation2.parameters:
                parameter_matchings.update(self.cosine_similarity(operation1_body, operation2_parameters, in_value="body to query"))
            if operation2.request_body:
                parameter_matchings.update(self.cosine_similarity(operation1_body, operation2_body, in_value="body to body"))
            if operation2.responses:
                parameter_matchings.update(self.cosine_similarity(operation1_body, operation2_responses, in_value="body to response"))

        for parameter, similarity in parameter_matchings.items():
            if similarity.similarity > self.threshold:
                similar_parameters[parameter] = similarity
            else:
                next_most_similar_parameters.append((parameter, similarity))

        return similar_parameters, next_most_similar_parameters


