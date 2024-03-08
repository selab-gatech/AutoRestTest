import os
from typing import List, Dict

from gensim.models import KeyedVectors
from gensim.downloader import load
from gensim.scripts.glove2word2vec import glove2word2vec

from scipy.spatial.distance import cosine

from .specification_parser import OperationProperties, SchemaProperties

def get_parameter_list(operation: OperationProperties) -> List[str]:
    if operation.parameters is None:
        return []
    return [parameter.lower().strip() for parameter, parameter_details in operation.parameters.items()]

def handle_properties(properties: Dict[str, SchemaProperties]) -> List[str]:
    property_list = []
    for item, item_details in properties.items():
        property_list.append(item.lower().strip())
    return property_list

def get_request_body_list(operation: OperationProperties) -> List[str]:
    if operation.request_body is None:
        return []
    request_body_list = []
    for request_body_type, request_body_properties in operation.request_body.items():
        # change handling to account for diff types other than properties
        if request_body_properties.properties:
            request_body_list += handle_properties(request_body_properties.properties)
    return request_body_list

def get_response_list(operation: OperationProperties) -> List[str]:
    if operation.responses is None:
        return []
    response_list = []
    for response_type, response_properties in operation.responses.items():
        if response_properties.content:
            for response, response_details in response_properties.content.items():
                if response_details.properties:
                    # change handling to account for diff types other than properties
                    response_list += handle_properties(response_details.properties)

    return response_list

class OperationDependencyComparator:
    def __init__(self):
        #current_file_path = os.path.dirname(os.path.abspath(__file__))
        #word_file = os.path.join(current_file_path, "models/glove.6B.50d.w2v.txt")
        #self.model = KeyedVectors.load_word2vec_format(word_file, binary=False)
        self.model = load("glove-wiki-gigaword-50")
        self.threshold = 0.65

    def cosine_similarity(self, operation1_parameters: List[str], operation2_responses: List[str]) -> Dict[str, str]:
        """
        Returns parameters that might map between operations
        """
        parameter_similarity = {}
        for parameter in operation1_parameters:
            for response in operation2_responses:
                if parameter in self.model and response in self.model:
                    similarity = 1 - cosine(self.model[parameter], self.model[response])
                    print(similarity, parameter, response)
                    if similarity > self.threshold:
                        parameter_similarity[parameter] = response

        return parameter_similarity

    def compare(self, operation1: OperationProperties, operation2: OperationProperties) -> Dict[str, str]:
        similar_parameters = {}

        operation2_responses = get_response_list(operation2)

        if operation1.parameters:
            operation1_parameters = get_parameter_list(operation1)
            similar_parameters = self.cosine_similarity(operation1_parameters, operation2_responses)

        if operation1.request_body:
            operation1_request_body = get_request_body_list(operation1)
            similar_request_body = self.cosine_similarity(operation1_request_body, operation2_responses)
            similar_parameters.update(similar_request_body)

        return similar_parameters

