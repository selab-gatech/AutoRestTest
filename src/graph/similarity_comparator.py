from typing import List, Dict, Tuple

from dataclasses import dataclass

from gensim.downloader import load

from scipy.spatial.distance import cosine

from src.graph.specification_parser import OperationProperties, SchemaProperties

@dataclass
class SimilarityValue:
    response_val: str = ""
    in_value: str = ""
    similarity: float = 0.0

    def to_dict(self):
        return {
            "response": self.response_val,
            "in_value": self.in_value,
            "similarity": self.similarity
        }

class OperationDependencyComparator:
    def __init__(self):
        #current_file_path = os.path.dirname(os.path.abspath(__file__))
        #word_file = os.path.join(current_file_path, "models/glove.6B.50d.w2v.txt")
        #self.model = KeyedVectors.load_word2vec_format(word_file, binary=False)
        self.model = load("glove-wiki-gigaword-50")
        self.threshold = 0.65

    @staticmethod
    def get_parameter_list(operation: OperationProperties) -> List[str]:
        if operation.parameters is None:
            return []
        return [parameter.lower().strip() for parameter, parameter_details in operation.parameters.items()]

    @staticmethod
    def handle_properties(properties: Dict[str, SchemaProperties]) -> List[str]:
        property_list = []
        for item, item_details in properties.items():
            property_list.append(item.lower().strip())
        return property_list

    @staticmethod
    def handle_schema_parameters(schema: SchemaProperties) -> List[str]:
        if schema.properties:
            return OperationDependencyComparator.handle_properties(schema.properties)
        elif schema.items:
            return OperationDependencyComparator.handle_schema_parameters(schema.items)
        return []

    @staticmethod
    def get_request_body_list(operation: OperationProperties) -> List[str]:
        if operation.request_body is None:
            return []
        request_body_list = []
        for request_body_type, request_body_properties in operation.request_body.items():
            request_body_list += OperationDependencyComparator.handle_schema_parameters(request_body_properties)
        return request_body_list

    @staticmethod
    def get_response_list(operation: OperationProperties) -> List[str]:
        if operation.responses is None:
            return []
        response_list = []
        for response_type, response_properties in operation.responses.items():
            if response_properties.content:
                for response, response_details in response_properties.content.items():
                    response_list += OperationDependencyComparator.handle_schema_parameters(response_details)
        return response_list

    def cosine_similarity(self, operation1_parameters: List[str], operation2_responses: List[str], in_value: str = None) -> Dict[str, SimilarityValue]:
        """
        Returns parameters that might map between operations
        """
        parameter_similarity = {}
        for parameter in operation1_parameters:
            for response in operation2_responses:
                if parameter in self.model and response in self.model:
                    similarity = 1 - cosine(self.model[parameter], self.model[response])
                    parameter_similarity[parameter] = SimilarityValue(
                        response_val=response,
                        in_value=in_value,
                        similarity=similarity
                    )

        return parameter_similarity

    def compare(self, operation1: OperationProperties, operation2: OperationProperties) -> (Dict[str, SimilarityValue], List[Tuple[str, SimilarityValue]]):
        parameter_matchings: Dict[str, SimilarityValue] = {}
        similar_parameters: Dict[str, SimilarityValue] = {}
        next_most_similar_parameters: List[(str, SimilarityValue)] = []
        operation2_responses = self.get_response_list(operation2)

        if operation1.parameters:
            operation1_parameters = self.get_parameter_list(operation1)
            parameter_matchings = self.cosine_similarity(operation1_parameters, operation2_responses, in_value="query")
        if operation1.request_body:
            operation1_parameters = self.get_request_body_list(operation1)
            parameter_matchings.update(self.cosine_similarity(operation1_parameters, operation2_responses, in_value="request body"))

        for parameter, similarity in parameter_matchings.items():
            if similarity.similarity > self.threshold:
                similar_parameters[parameter] = similarity
            else:
                next_most_similar_parameters.append((parameter, similarity))

        return similar_parameters, next_most_similar_parameters

