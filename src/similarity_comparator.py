from typing import List, Dict

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

    def cosine_similarity(self, operation1_parameters: List[str], operation2_responses: List[str]):

        #operation1_parameters = ["Test1", "Test2", "Tset3"]
        #operation2_responses = ["Test3", "Tes1", "Test2"]

        tfidf_matrix = self.vectorizer.fit_transform(operation1_parameters + operation2_responses)

        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        parameter_similarity = {}
        for i in range(len(similarity_matrix)):
            for j in range(len(similarity_matrix[i])):
                #if i < len(operation1_parameters) <= j < len(operation2_responses) + len(operation1_parameters):
                #    print(similarity_matrix[i][j])
                if i < len(operation1_parameters) <= j < len(operation2_responses) + len(operation1_parameters) and similarity_matrix[i][j] > 0.7:
                    parameter_similarity[operation1_parameters[i]] = operation2_responses[j - len(operation1_parameters)]
        return parameter_similarity

    def compare(self, operation1: OperationProperties, operation2: OperationProperties) -> (Dict, Dict):
        operation1_parameters = get_parameter_list(operation1)
        operation2_responses = get_response_list(operation2)

        similarty_1to2 = self.cosine_similarity(operation1_parameters, operation2_responses)

        operation2_parameters = get_parameter_list(operation2)
        operation1_responses = get_response_list(operation1)

        similarity_2to1 = self.cosine_similarity(operation2_parameters, operation1_responses)

        return similarty_1to2, similarity_2to1

