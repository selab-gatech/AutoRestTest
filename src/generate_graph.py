from typing import List, Dict

from .specification_parser import OperationProperties, SpecificationParser
from .similarity_comparator import OperationDependencyComparator, SimilarityValue


class OperationNode:
    def __init__(self, operation_properties: OperationProperties):
        self.operation_id = operation_properties.operation_id
        self.operation_properties: OperationProperties = operation_properties
        self.outgoing_edges: List[OperationEdge] = []
        self.tentative_edges: List[(str, SimilarityValue)] = [] # stored as decreasing list of similarity values

class OperationEdge:
    def __init__(self, source: OperationNode, destination: OperationNode, similar_parameters: Dict[str, SimilarityValue]=None):
        if similar_parameters is None:
            similar_parameters = {}
        self.source: OperationNode = source
        self.destination: OperationNode = destination
        self.similar_parameters: Dict[str, SimilarityValue] = similar_parameters # have parameters as the key (similarity value has response and in_value)

class OperationGraph:
    def __init__(self, spec_path, spec_name=None):
        self.spec_path = spec_path
        self.spec_name = spec_name
        self.operation_nodes: Dict[str, OperationNode] = {}
        self.operation_edges: List[OperationEdge] = []

    def add_operation_node(self, operation_properties: OperationProperties):
        self.operation_nodes[operation_properties.operation_id] = OperationNode(operation_properties)

    def add_operation_edge(self, operation_id: str, dependent_operation_id: str, parameters: Dict[str, SimilarityValue]):
        if operation_id not in self.operation_nodes:
            raise ValueError(f"Operation {operation_id} not found in the graph")
        source_node = self.operation_nodes[operation_id]
        destination_node = self.operation_nodes[dependent_operation_id]
        edge = OperationEdge(source=source_node, destination=destination_node, similar_parameters=parameters)
        self.operation_edges.append(edge)
        source_node.outgoing_edges.append(edge)

    def update_operation_dependencies(self, operation_id: str, dependent_operation_id: str, parameters: Dict[str, SimilarityValue], next_closest_similarities: List[(int, (str, SimilarityValue))]):
        if operation_id not in self.operation_nodes:
            raise ValueError(f"Operation {operation_id} not found in the graph")
        source_node = self.operation_nodes[operation_id]
        if len(parameters) > 0:
            self.add_operation_edge(operation_id, dependent_operation_id, parameters)
        if len(next_closest_similarities) > 0:
            source_node.tentative_edges = next_closest_similarities

    def determine_dependencies(self, operations):
        operation_dependency_comparator = OperationDependencyComparator()
        for operation_id, operation_properties in operations.items():
            for dependent_operation_id, dependent_operation_properties in operations.items():
                # Note: We consider responses from get requests as dependencies for request bodies
                # Note: We consider responses from post/put requests as dependencies for get requests
                if (operation_properties.similar_parameters or operation_properties.request_body) and dependent_operation_properties.responses:
                    parameter_similarities, next_closest_similarities = operation_dependency_comparator.compare(operation_properties, dependent_operation_properties)
                    self.update_operation_dependencies(operation_id, dependent_operation_id, parameter_similarities, next_closest_similarities)

                    #if len(similarity_2to1) > 0:
                    #    self.add_operation_edge(dependent_operation_id, operation_id, similarity_2to1)

    def create_graph(self):
        spec_parser = SpecificationParser(self.spec_path, self.spec_name)

        operations: Dict[str, OperationProperties] = spec_parser.parse_specification()
        for operation_id, operation_properties in operations.items():
            self.add_operation_node(operation_properties)

        self.determine_dependencies(operations)

if __name__ == "__main__":
    #operation_graph = OperationGraph(spec_path="specs/original/oas/genome-nexus.yaml", spec_name="genome-nexus")
    operation_graph = OperationGraph(spec_path="specs/original/oas/genome-nexus.yaml", spec_name="genome-nexus")
    operation_graph.create_graph()
    for operation_id, operation_node in operation_graph.operation_nodes.items():
        print(f"Operation: {operation_id}")
    for operation_edge in operation_graph.operation_edges:
        print(f"Edge: {operation_edge.source.operation_id} -> {operation_edge.destination.operation_id} with parameters: {operation_edge.similar_parameters}")