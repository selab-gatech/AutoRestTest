from src.generate_graph import OperationGraph

if __name__ == "__main__":
    operation_graph = OperationGraph(spec_path="specs/original/oas/genome-nexus.yaml", spec_name="genome-nexus")
    operation_graph.create_graph()
    for operation_id, operation_node in operation_graph.operation_nodes.items():
        print(f"Operation: {operation_id}")
    for operation_edge in operation_graph.operation_edges:
        print(
            f"Edge: {operation_edge.source.operation_id} -> {operation_edge.destination.operation_id} with parameters: {operation_edge.parameters}")