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
        self.threshold = 0.8

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
    def handle_response_params(response: SchemaProperties, response_params: List[Dict[str,str]]) -> None:
        if response.properties:
            for item, item_details in response.properties.items():
                if {OperationDependencyComparator.handle_parameter_cases(item): item} not in response_params:
                    response_params.append({OperationDependencyComparator.handle_parameter_cases(item): item})
                OperationDependencyComparator.handle_response_params(item_details, response_params)
        elif response.items:
            OperationDependencyComparator.handle_response_params(response.items, response_params)
        else:
            return

    @staticmethod
    def handle_body_params(body: SchemaProperties) -> List[Dict[str,str]]:
        object_params = []
        if body.properties:
            object_params = [{OperationDependencyComparator.handle_parameter_cases(item): item} for item, item_details in body.properties.items()]
        elif body.items:
            object_params = OperationDependencyComparator.handle_body_params(body.items)
        return object_params

    @staticmethod
    def get_request_body_list(operation: OperationProperties) -> List[Dict[str,str]]:
        if operation.request_body is None:
            return []
        request_body_list = []
        for request_body_type, request_body_properties in operation.request_body.items():
            request_body_list += OperationDependencyComparator.handle_body_params(request_body_properties)
        return request_body_list

    @staticmethod
    def get_response_list(operation: OperationProperties) -> List[Dict[str,str]]:
        if operation.responses is None:
            return []
        response_list = []
        for status_code, response_properties in operation.responses.items():
            if status_code and status_code[0] == "2" and response_properties.content:
                for response, response_details in response_properties.content.items():
                    curr_responses = []
                    OperationDependencyComparator.handle_response_params(response_details, curr_responses)
                    response_list += curr_responses
        return response_list

    def encode_sentence_or_word(self, thing: str):
        words = thing.split(" ")
        word_vectors = [self.model[word] for word in words if word in self.model]
        return np.mean(word_vectors, axis=0) if word_vectors else None

    def cosine_similarity(self, operation1_vals: List[Dict[str,str]], operation2_vals: List[Dict[str,str]], in_value: str = None) -> Dict[str, List[SimilarityValue]]:
        """
        Returns parameters that might map between operations
        """
        param_param_similarity = {}
        for parameter_pairing in operation1_vals:
            for processed_parameter, parameter in parameter_pairing.items():
                param_param_similarity[parameter] = []
                for dependency_pairing in operation2_vals:
                    for processed_dependency, dependency in dependency_pairing.items():
                        parameter_encoding = self.encode_sentence_or_word(processed_parameter)
                        dependency_encoding = self.encode_sentence_or_word(processed_dependency)
                        if parameter_encoding is not None and dependency_encoding is not None:
                            similarity = 1 - cosine(parameter_encoding, dependency_encoding)
                            param_param_similarity[parameter].append(SimilarityValue(
                                dependent_val=dependency,
                                in_value=in_value,
                                similarity=similarity
                            ))

        return param_param_similarity

    def compare_cosine(self, operation1: OperationProperties, operation2: OperationProperties) -> (Dict[str, List[SimilarityValue]], List[Tuple[str, SimilarityValue]]):
        parameter_matchings: Dict[str, List[SimilarityValue]] = {}
        similar_parameters: Dict[str, List[SimilarityValue]] = {}
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
                added_parameter_matchings = self.cosine_similarity(operation1_parameters, operation2_body, in_value="query to body")
                for parameter, similarities in added_parameter_matchings.items():
                    parameter_matchings.setdefault(parameter, []).extend(similarities)
            if operation2.responses:
                added_parameter_matchings = self.cosine_similarity(operation1_parameters, operation2_responses, in_value="query to response")
                for parameter, similarities in added_parameter_matchings.items():
                    parameter_matchings.setdefault(parameter, []).extend(similarities)

        if operation1.request_body:
            if operation2.parameters:
                added_parameter_matchings = self.cosine_similarity(operation1_body, operation2_parameters, in_value="body to query")
                for parameter, similarities in added_parameter_matchings.items():
                    parameter_matchings.setdefault(parameter, []).extend(similarities)
            if operation2.request_body:
                added_parameter_matchings = self.cosine_similarity(operation1_body, operation2_body, in_value="body to body")
                for parameter, similarities in added_parameter_matchings.items():
                    parameter_matchings.setdefault(parameter, []).extend(similarities)
            if operation2.responses:
                added_parameter_matchings = self.cosine_similarity(operation1_body, operation2_responses, in_value="body to response")
                for parameter, similarities in added_parameter_matchings.items():
                    parameter_matchings.setdefault(parameter, []).extend(similarities)

        for parameter, similarities in parameter_matchings.items():
            for similarity in similarities:
                if parameter not in similar_parameters:
                    similar_parameters[parameter] = []
                if similarity.similarity > self.threshold:
                    similar_parameters[parameter].append(similarity)
                else:
                    # TODO: Update tentative parameter handling
                    next_most_similar_parameters.append((parameter, similarity))

        return similar_parameters, next_most_similar_parameters


