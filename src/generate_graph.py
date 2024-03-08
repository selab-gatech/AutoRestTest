from typing import List, Dict

from .specification_parser import OperationProperties, SpecificationParser
from .similarity_comparator import OperationDependencyComparator

class OperationNode:
    def __init__(self, operation_properties: OperationProperties):
        self.operation_id = operation_properties.operation_id
        self.operation_properties: OperationProperties = operation_properties
        self.adjacent_nodes: Dict[str, OperationProperties] = {}

    def add_adjacent_node(self, operation_properties: OperationProperties):
        self.adjacent_nodes.setdefault(operation_properties.operation_id, operation_properties)

class OperationGraph:
    def __init__(self, spec_path, spec_name=None):
        self.spec_path = spec_path
        self.spec_name = spec_name
        self.operation_nodes: Dict[str, OperationNode] = {}

    def add_operation_node(self, operation_properties: OperationProperties):
        self.operation_nodes[operation_properties.operation_id] = OperationNode(operation_properties)

    def add_dependency(self, operation_id: str, dependent_operation_id: str):
        if operation_id not in self.operation_nodes:
            raise ValueError(f"Operation {operation_id} not found in the graph")
        self.operation_nodes[operation_id].add_adjacent_node(self.operation_nodes[dependent_operation_id].operation_properties)

    def determine_dependencies(self, operations):
        operation_dependency_comparator = OperationDependencyComparator()
        visited = set()
        for operation_id, operation_properties in operations.items():
            visited.add(operation_id)
            for dependent_operation_id, dependent_operation_properties in operations.items():
                if dependent_operation_id in visited:
                    continue
                similarity_1to2, similarity_2to1 = operation_dependency_comparator.compare(operation_properties, dependent_operation_properties)
                if similarity_1to2 > 0.7:
                    self.add_dependency(operation_id, dependent_operation_id)
                if similarity_2to1 > 0.7:
                    self.add_dependency(dependent_operation_id, operation_id)
    def create_graph(self):
        spec_parser = SpecificationParser(self.spec_path, self.spec_name)
        operations: Dict[str, OperationProperties] = spec_parser.parse_specification()
        for operation_id, operation_properties in operations.items():
            self.add_operation_node(operation_properties)
        self.determine_dependencies(operations)

if __name__ == "__main__":
    #operation_graph = OperationGraph(spec_path="specs/original/oas/genome-nexus.yaml", spec_name="genome-nexus")
    operation_graph = OperationGraph(spec_path="specs/original/oas/fdic.yaml", spec_name="fdic")
    operation_graph.create_graph()
    for operation_id, operation_node in operation_graph.operation_nodes.items():
        print(f"Operation: {operation_id}")
        print(f"Adjacent Nodes: {operation_node.adjacent_nodes.keys()}")
        print()