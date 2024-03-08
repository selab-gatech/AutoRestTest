from typing import List, Dict

from gensim.models import KeyedVectors
from gensim.downloader import load

from scipy.spatial.distance import cosine

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
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2,5))
        self.model = load("glove-wiki-gigaword-50")
        self.threshold = 0.65

    def cosine_similarity(self, operation1_parameters: List[str], operation2_responses: List[str]):

        #operation1_parameters = ["apples", "numbers", "testing"]
        #operation2_responses = ["testing2", "apple", "in_numbers"]

        #tfidf_matrix = self.vectorizer.fit_transform(operation1_parameters + operation2_responses)

        #similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        #parameter_similarity = {}
        #for i in range(len(similarity_matrix)):
        #    for j in range(len(similarity_matrix[i])):
        #        if i < len(operation1_parameters) <= j < len(operation2_responses) + len(operation1_parameters):
        #            print(similarity_matrix[i][j])
        #        if i < len(operation1_parameters) <= j < len(operation2_responses) + len(operation1_parameters) and similarity_matrix[i][j] > self.threshold:
        #            parameter_similarity[operation1_parameters[i]] = operation2_responses[j - len(operation1_parameters)]

        parameter_similarity = {}
        for parameter in operation1_parameters:
            for response in operation2_responses:
                if parameter in self.model and response in self.model:
                    similarity = 1 - cosine(self.model[parameter], self.model[response])
                    print(similarity, parameter, response)
                    if similarity > self.threshold:
                        parameter_similarity[parameter] = response

        return parameter_similarity

    def compare(self, operation1: OperationProperties, operation2: OperationProperties) -> (Dict, Dict):
        operation1_parameters = get_parameter_list(operation1)
        operation2_responses = get_response_list(operation2)

        similarty_1to2 = self.cosine_similarity(operation1_parameters, operation2_responses)

        operation2_parameters = get_parameter_list(operation2)
        operation1_responses = get_response_list(operation1)

        similarity_2to1 = self.cosine_similarity(operation2_parameters, operation1_responses)

        return similarty_1to2, similarity_2to1

