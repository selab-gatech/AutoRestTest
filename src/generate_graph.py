import heapq
import logging
from typing import List, Dict, Tuple, Optional

from src.request_generator import RequestGenerator
from src.graph.specification_parser import OperationProperties, SpecificationParser
from src.graph.similarity_comparator import OperationDependencyComparator, SimilarityValue
from src.utils import EmbeddingModel


class OperationNode:
    def __init__(self, operation_properties: OperationProperties):
        self.operation_id = operation_properties.operation_id
        self.operation_properties: OperationProperties = operation_properties
        self.outgoing_edges: List[OperationEdge] = []
        self.tentative_edges: List[OperationEdge] = []

class OperationEdge:
    def __init__(self, source: OperationNode, destination: OperationNode, similar_parameters: Dict[str, List[SimilarityValue]]=None):
        if similar_parameters is None:
            similar_parameters = {}
        self.source: OperationNode = source
        self.destination: OperationNode = destination
        self.similar_parameters: Dict[str, List[SimilarityValue]] = similar_parameters # have parameters as the key (similarity value has response param and in_value)

class OperationGraph:
    def __init__(self, spec_path, spec_name=None, spec_parser: SpecificationParser = None, embedding_model=None):
        self.spec_path = spec_path
        self.spec_name = spec_name
        self.spec_parser = spec_parser
        self.request_generator: Optional[RequestGenerator] = None
        self.operation_nodes: Dict[str, OperationNode] = {}
        self.operation_edges: List[OperationEdge] = []
        self.next_most_similar_count = 3
        self.embedding_model: EmbeddingModel = embedding_model
        self.dependency_comparator = OperationDependencyComparator(model=embedding_model)

    def print_graph(self):
        for operation_id, operation_node in self.operation_nodes.items():
            print("=====================================")
            print(f"Operation: {operation_id}")
            #print(f"Operation Properties: {operation_node.operation_properties}")
            for edge in operation_node.outgoing_edges:
                print(f"Edge: {edge.source.operation_id} -> {edge.destination.operation_id} with parameters: {edge.similar_parameters}")
            for tentative_edge in operation_node.tentative_edges:
                print(f"Tentative Edge: {tentative_edge.source.operation_id} -> {tentative_edge.destination.operation_id} with parameters: {tentative_edge.similar_parameters}")
            print()
            print()

    def print_edges(self):
        for operation_edge in self.operation_edges:
            print(f"Edge: {operation_edge.source.operation_id} -> {operation_edge.destination.operation_id} with parameters: {operation_edge.similar_parameters}")

    def add_operation_node(self, operation_properties: OperationProperties):
        self.operation_nodes[operation_properties.operation_id] = OperationNode(operation_properties)

    def assign_request_generator(self, request_generator: RequestGenerator):
        self.request_generator = request_generator

    def add_operation_edge(self, operation_id: str, dependent_operation_id: str, parameters: Dict[str, List[SimilarityValue]]):
        if operation_id not in self.operation_nodes:
            raise ValueError(f"Operation {operation_id} not found in the graph")
        source_node = self.operation_nodes[operation_id]
        destination_node = self.operation_nodes[dependent_operation_id]
        edge = OperationEdge(source=source_node, destination=destination_node, similar_parameters=parameters)
        self.operation_edges.append(edge)
        source_node.outgoing_edges.append(edge)
        #print(f"Added edge from {operation_id} to {dependent_operation_id} with parameters: {parameters}")

    def add_tentative_edge(self, operation_id: str, dependent_operation_id: str, next_closest_similarities: List[Tuple[str, SimilarityValue]]):
        # TODO: Update tentative edge handling for lists
        if operation_id not in self.operation_nodes:
            raise ValueError(f"Operation {operation_id} not found in the graph")
        source_node = self.operation_nodes[operation_id]
        destination_node = self.operation_nodes[dependent_operation_id]
        similar_parameters = {}
        # recall that next_closest_similarities is a list matching params in operation to params in dependent_operation
        for next_closest_similarity in next_closest_similarities:
            if next_closest_similarity[0] not in similar_parameters:
                similar_parameters[next_closest_similarity[0]] = []
            similar_parameters[next_closest_similarity[0]].append(next_closest_similarity[1])

        edge = OperationEdge(source=source_node, destination=destination_node, similar_parameters=similar_parameters)
        source_node.tentative_edges.append(edge)
        source_node.tentative_edges = heapq.nlargest(self.next_most_similar_count, source_node.tentative_edges,
                                                            key=lambda x: next(iter(x.similar_parameters.values())).similarity)  # small n so efficient

    def update_operation_dependencies(self, operation_id: str, dependent_operation_id: str, similar_parameters: Dict[str, List[SimilarityValue]], next_closest_similarities: List[Tuple[str, SimilarityValue]]):
        if operation_id not in self.operation_nodes:
            raise ValueError(f"Operation {operation_id} not found in the graph")
        if len(similar_parameters) > 0:
            self.add_operation_edge(operation_id, dependent_operation_id, similar_parameters)
        # Only add tentative edges if there are no similar parameters
        elif len(next_closest_similarities) > 0:
            # TODO: Update tentative edge handling for lists
            self.add_tentative_edge(operation_id, dependent_operation_id, next_closest_similarities)

    def remove_edge(self, operation_id: str, dependent_operation_id: str):
        if operation_id not in self.operation_nodes:
            raise ValueError(f"Operation {operation_id} not found in the graph")
        source_node = self.operation_nodes[operation_id]
        for edge in source_node.outgoing_edges:
            if edge.destination.operation_id == dependent_operation_id:
                source_node.outgoing_edges.remove(edge)
                self.operation_edges.remove(edge)
                return

    def determine_dependencies(self, operations):
        for operation_id, operation_properties in operations.items():
            for dependent_operation_id, dependent_operation_properties in operations.items():
                if operation_id == dependent_operation_id:
                    continue
                # Note: We consider responses from get requests as dependencies for request bodies
                # Note: We consider responses from post/put requests as dependencies for get requests
                parameter_similarities, next_closest_similarities = self.dependency_comparator.compare_cosine(operation_properties, dependent_operation_properties)
                self.update_operation_dependencies(operation_id, dependent_operation_id, parameter_similarities, next_closest_similarities)
            if self.operation_nodes[operation_id].tentative_edges and not self.operation_nodes[operation_id].outgoing_edges:
                # Assign top tentative edges to outgoing edges if there are no similar parameters
                self.operation_nodes[operation_id].outgoing_edges = self.operation_nodes[operation_id].tentative_edges

    def create_graph(self, auto_validate=True):
        operations: Dict[str, OperationProperties] = self.spec_parser.parse_specification()
        print("PARSED SPECIFICATION!!!")
        for operation_id, operation_properties in operations.items():
            self.add_operation_node(operation_properties)
        self.determine_dependencies(operations)
        print("COMPLETED COSINE SIMILARITY!!!")


if __name__ == "__main__":
    operation_graph = OperationGraph(spec_path="specs/original/oas/genome-nexus.yaml", spec_name="genome-nexus", initialize_graph=False)
    #operation_graph = OperationGraph(spec_path="specs/original/oas/ocvn.yaml", spec_name="ocvn")
    operation_graph.create_graph()
    for operation_id, operation_node in operation_graph.operation_nodes.items():
        print("=====================================")
        print(f"Operation: {operation_id}")
        for edge in operation_node.outgoing_edges:
            print(f"Edge: {edge.source.operation_id} -> {edge.destination.operation_id} with parameters: {edge.similar_parameters}")
        for tentative_edge in operation_node.tentative_edges:
            print(f"Tentative Edge: {tentative_edge.source.operation_id} -> {tentative_edge.destination.operation_id} with parameters: {tentative_edge.similar_parameters}")
        print()
        print()

    #for operation_edge in operation_graph.operation_edges:
    #    print(f"Edge: {operation_edge.source.operation_id} -> {operation_edge.destination.operation_id} with parameters: {operation_edge.similar_parameters}")