import os
from typing import List, Dict

from gensim.models import KeyedVectors
from gensim.downloader import load
from gensim.scripts.glove2word2vec import glove2word2vec

from scipy.spatial.distance import cosine

from .specification_parser import OperationProperties

def get_parameter_list(operation: OperationProperties) -> List[str]:
    if operation.parameters is None:
        return []
    return [parameter.lower().strip() for parameter, parameter_details in operation.parameters.items()]

def get_response_list(operation: OperationProperties) -> List[str]:
    if operation.responses is None:
        return []
    response_list = []
    for response_type, response_properties in operation.responses.items():
        if response_properties.content:
            for response, response_details in response_properties.content.items():
                if response_details.properties:
                    for item, item_details in response_details.properties.items():
                        response_list.append(item.lower().strip())
    return response_list

class OperationDependencyComparator:
    def __init__(self):
        current_file_path = os.path.dirname(os.path.abspath(__file__))
        #glove_file = os.path.join(current_file_path, "models/glove.6B.50d.txt")
        word_file = os.path.join(current_file_path, "models/glove.6B.50d.w2v.txt")
        #glove2word2vec(glove_file, tmp_file)
        self.model = KeyedVectors.load_word2vec_format(word_file, binary=False)
        self.threshold = 0.65

    def cosine_similarity(self, operation1_parameters: List[str], operation2_responses: List[str]):
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
        operation1_parameters = get_parameter_list(operation1)
        operation2_responses = get_response_list(operation2)

        similar_parameters = self.cosine_similarity(operation1_parameters, operation2_responses)

        #operation2_parameters = get_parameter_list(operation2)
        #operation1_responses = get_response_list(operation1)

        #similarity_2to1 = self.cosine_similarity(operation2_parameters, operation1_responses)

        return similar_parameters

