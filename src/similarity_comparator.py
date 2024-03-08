from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .specification_parser import OperationProperties

def get_parameter_list(operation: OperationProperties) -> List[str]:
    if operation.parameters is None:
        return []
    return [parameter.lower().strip() for parameter, parameter_details in operation.parameters.items()]

def get_response_list(operation: OperationProperties) -> List[str]:
    if operation.responses is None:
        return []
    return [response.lower().strip() for response, response_details in operation.responses.items()]

class OperationDependencyComparator:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def cosine_similarity(self, operation1_parameters, operation2_responses):

        parameter_text = " ".join(operation1_parameters)
        response_text = " ".join(operation2_responses)

        tfidf_matrix = self.vectorizer.fit_transform([parameter_text, response_text])

        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

        return similarity_matrix[0]

    def compare(self, operation1: OperationProperties, operation2: OperationProperties):
        operation1_parameters = get_parameter_list(operation1)
        operation2_responses = get_response_list(operation2)

        similarty_1to2 = self.cosine_similarity(operation1_parameters, operation2_responses)

        operation2_parameters = get_parameter_list(operation2)
        operation1_responses = get_response_list(operation1)

        similarity_2to1 = self.cosine_similarity(operation2_parameters, operation1_responses)

        return similarty_1to2, similarity_2to1

