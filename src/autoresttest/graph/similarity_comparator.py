import logging
import os
from typing import Dict, List, Tuple

import numpy as np
from gensim.downloader import load

from scipy.spatial.distance import cosine

from autoresttest.models import (
    OperationProperties,
    ParameterKey,
    SchemaProperties,
    SimilarityValue,
)

from autoresttest.utils import EmbeddingModel


class OperationDependencyComparator:
    def __init__(self, model: EmbeddingModel):
        self.model = model
        self.threshold = 0.8

    def get_parameter_list(
        self, operation: OperationProperties
    ) -> List[Tuple[str, ParameterKey, str]]:
        if operation.parameters is None:
            return []
        parameter_list: List[Tuple[str, ParameterKey, str]] = []
        for parameter_key, parameter_details in operation.parameters.items():
            processed = self.model.handle_word_cases(parameter_details.name)
            location = parameter_key[1] if isinstance(parameter_key, tuple) else None
            parameter_list.append(
                (
                    processed,
                    parameter_key,
                    location or parameter_details.in_value or "query",
                )
            )
        return parameter_list

    def handle_response_params(
        self, response: SchemaProperties, response_params: List[Dict[str, str]]
    ) -> None:
        if response.properties:
            for item, item_details in response.properties.items():
                if {self.model.handle_word_cases(item): item} not in response_params:
                    response_params.append({self.model.handle_word_cases(item): item})
                self.handle_response_params(item_details, response_params)
        elif response.items:
            self.handle_response_params(response.items, response_params)
        else:
            return

    def handle_body_params(self, body: SchemaProperties) -> List[Tuple[str, str]]:
        object_params: List[Tuple[str, str]] = []
        if body.properties:
            object_params = [
                (self.model.handle_word_cases(item), item)
                for item, item_details in body.properties.items()
            ]
        elif body.items:
            object_params = self.handle_body_params(body.items)
        return object_params

    def get_request_body_list(
        self, operation: OperationProperties
    ) -> List[Tuple[str, str, str]]:
        if operation.request_body is None:
            return []
        request_body_list = []
        for (
            request_body_type,
            request_body_properties,
        ) in operation.request_body.items():
            request_body_list += [
                (processed, item, "body")
                for processed, item in self.handle_body_params(request_body_properties)
            ]
        return request_body_list

    def get_response_list(
        self, operation: OperationProperties
    ) -> List[Tuple[str, str, str]]:
        if operation.responses is None:
            return []
        response_list = []
        for status_code, response_properties in operation.responses.items():
            if status_code and status_code[0] == "2" and response_properties.content:
                for response, response_details in response_properties.content.items():
                    curr_responses = []
                    self.handle_response_params(response_details, curr_responses)
                    response_list += [
                        (processed, item, "response")
                        for processed_item in curr_responses
                        for processed, item in processed_item.items()
                    ]
        return response_list

    def cosine_similarity(
        self,
        operation1_vals: List[Tuple[str, ParameterKey, str]],
        operation2_vals: List[Tuple[str, object, str]],
    ) -> Dict[ParameterKey, List[SimilarityValue]]:
        """
        Returns parameters that might map between operations
        """
        param_param_similarity = {}
        for processed_parameter, parameter_key, parameter_loc in operation1_vals:
            param_param_similarity.setdefault(parameter_key, [])
            for processed_dependency, dependency_key, dependency_loc in operation2_vals:
                param_embedding = self.model.encode_sentence_or_word(
                    processed_parameter
                )
                dependency_embedding = self.model.encode_sentence_or_word(
                    processed_dependency
                )
                if param_embedding is not None and dependency_embedding is not None:
                    similarity = 1 - cosine(param_embedding, dependency_embedding)
                    param_param_similarity[parameter_key].append(
                        SimilarityValue(
                            dependent_val=dependency_key,
                            in_value=f"{parameter_loc} to {dependency_loc}",
                            similarity=similarity,
                        )
                    )

        return param_param_similarity

    def compare_cosine(
        self, operation1: OperationProperties, operation2: OperationProperties
    ) -> Tuple[
        Dict[ParameterKey, List[SimilarityValue]], List[Tuple[ParameterKey, SimilarityValue]]
    ]:
        parameter_matchings: Dict[ParameterKey, List[SimilarityValue]] = {}
        similar_parameters: Dict[ParameterKey, List[SimilarityValue]] = {}
        next_most_similar_parameters: List[Tuple[ParameterKey, SimilarityValue]] = []

        operation1_parameters = self.get_parameter_list(operation1)
        operation1_body = self.get_request_body_list(operation1)
        operation2_parameters = self.get_parameter_list(operation2)
        operation2_body = self.get_request_body_list(operation2)
        operation2_responses = self.get_response_list(operation2)

        if operation1.parameters:
            if operation2.parameters:
                parameter_matchings = self.cosine_similarity(
                    operation1_parameters, operation2_parameters
                )
            if operation2.request_body:
                added_parameter_matchings = self.cosine_similarity(
                    operation1_parameters, operation2_body
                )
                for parameter, similarities in added_parameter_matchings.items():
                    parameter_matchings.setdefault(parameter, []).extend(similarities)
            if operation2.responses:
                added_parameter_matchings = self.cosine_similarity(
                    operation1_parameters, operation2_responses
                )
                for parameter, similarities in added_parameter_matchings.items():
                    parameter_matchings.setdefault(parameter, []).extend(similarities)

        if operation1.request_body:
            if operation2.parameters:
                added_parameter_matchings = self.cosine_similarity(
                    operation1_body, operation2_parameters
                )
                for parameter, similarities in added_parameter_matchings.items():
                    parameter_matchings.setdefault(parameter, []).extend(similarities)
            if operation2.request_body:
                added_parameter_matchings = self.cosine_similarity(
                    operation1_body, operation2_body
                )
                for parameter, similarities in added_parameter_matchings.items():
                    parameter_matchings.setdefault(parameter, []).extend(similarities)
            if operation2.responses:
                added_parameter_matchings = self.cosine_similarity(
                    operation1_body, operation2_responses
                )
                for parameter, similarities in added_parameter_matchings.items():
                    parameter_matchings.setdefault(parameter, []).extend(similarities)

        for parameter, similarities in parameter_matchings.items():
            for similarity in similarities:
                if parameter not in similar_parameters:
                    similar_parameters[parameter] = []
                if similarity.similarity > self.threshold:
                    similar_parameters[parameter].append(similarity)
                else:
                    next_most_similar_parameters.append((parameter, similarity))

        return similar_parameters, next_most_similar_parameters
