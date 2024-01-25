import json
import re
import numpy as np
from graphviz import Digraph

from .sequence import Sequence
from .sequence import SequenceOrigin


class Path:
    def __init__(self):
        self.nodes = []

    def duplicate(self):
        path = Path()
        path.nodes = list(self.nodes)
        return path

    def append(self, elem):
        self.nodes.append(elem)

    def __iter__(self):
        return self.nodes.__iter__()

    def __getitem__(self, item):
        return self.nodes[item]

    def __str__(self):
        if len(self.nodes):
            return ''
        return ' -> '.join(self.nodes)

    def __hash__(self):
        return hash(self.__str__())

    def remove(self, elem):
        self.nodes.remove(elem)

    def __len__(self):
        return len(self.nodes)

    def __eq__(self, other):
        if len(other.nodes) != len(self.nodes) or not isinstance(other, Path):
            return False
        return str(other) == str(self)

    def pop(self, ind):
        self.nodes.pop(ind)


class Edge:
    def __init__(self, from_node, to_node, name):
        self.from_node = from_node
        self.to_node = to_node
        self.name = name

    def __str__(self):
        return f'{self.from_node}->{self.to_node}'


class OperationDependencyGraph:
    def __init__(self):
        self.nodes = set()
        self.edges = set()
        self.from_to_dict = {}
        self.feed_from_edges = {}
        self.output_to_edges = {}
        self.dependency_path = '../dependency.json'
        self.threshold_length = 200

    def add_edge(self, from_node, to_node, name):
        if self.from_to_dict.__contains__((from_node, to_node)):
            return self.from_to_dict[(from_node, to_node)]
        edge = Edge(from_node, to_node, name)
        feed_from_edges = self.feed_from_edges.get(to_node, [])
        output_to_edges = self.output_to_edges.get(from_node, [])
        feed_from_edges.append(edge)
        output_to_edges.append(edge)
        self.feed_from_edges[to_node] = feed_from_edges
        self.output_to_edges[from_node] = output_to_edges
        self.from_to_dict[(from_node, to_node)] = edge
        self.nodes.add(from_node)
        self.nodes.add(to_node)
        self.edges.add(edge)
        return edge

    def remove_path_variable(self, pattern, target):
        notations = pattern.findall(target)
        for notation in notations:
            target = target.replace(notation, "")
        return target

    def load_traffic_dependency(self, path):
        with open(path, 'r', encoding="utf-8") as jsonfile:
            data = jsonfile.read()
            obj = json.loads(data)
            return obj

    def get_traffic_yaml_mapped_methods_map(self, dependency, yaml_method_map):
        path_pattern = re.compile('\{(.*?)\}')
        results = set()
        for method in dependency:
            nominal_name = self.remove_path_variable(path_pattern, method)
            if nominal_name in yaml_method_map:
                to_method = yaml_method_map[nominal_name]
                feed_from_method_dependency = dependency[method]
                for feed_from_method in feed_from_method_dependency:
                    nominal_feed_from_name = self.remove_path_variable(path_pattern, feed_from_method)
                    if nominal_feed_from_name in yaml_method_map:
                        from_method = yaml_method_map[nominal_feed_from_name]
                        to_method.dependency_from_traffic[from_method] = feed_from_method_dependency[feed_from_method]
                        from_method.output_to_method.add(to_method)
                        to_method.feed_from_method.add(from_method)
                        results.add((from_method, to_method))
        return results

    def get_traffic_map_with_yaml(self, path):
        dependency = self.load_traffic_dependency(path)
        yaml_method_map = self.get_yaml_method_path_map()
        results = self.get_traffic_yaml_mapped_methods_map(dependency, yaml_method_map)
        print(len(results), results)

    def get_yaml_method_path_map(self):
        path_pattern = re.compile('\{(.*?)\}')
        result = {}
        for method in self.nodes:
            result[str(method.method_type).upper() + "-" + self.remove_path_variable(path_pattern,
                                                                                     method.method_path)] = method
        return result

    def add_node(self, node):
        return self.nodes.add(node)

    def get_output_edges(self, from_node):
        return self.output_to_edges.get(from_node, [])

    def get_feed_from_edges(self, to_node):
        return self.feed_from_edges.get(to_node, [])

    def draw(self, path='graph.txt'):
        data_graph = Digraph()
        for node in self.nodes:
            data_graph.node(f'{node.method_type.upper()} {node.method_path}')
        for edge in self.edges:
            from_node = edge.from_node
            to_node = edge.to_node
            data_graph.edge(f'{to_node.method_type.upper()} {to_node.method_path}',
                            f'{from_node.method_type.upper()} {from_node.method_path}', edge.name)
        # data_graph.render(path)
        with open(path, 'w+') as output_graph:
            output_graph.writelines(data_graph.source)

    def generate_graph_sequence(self, method):
        paths = []
        covered_apis = set()
        def traverse_path_recursive(method, path):
            if method in path:
                paths.append(path.duplicate())
            elif len(method.output_to_method) == 0:
                tmp = path.duplicate()
                tmp.append(method)
                covered_apis.add(method)
                paths.append(tmp)
            else:
                for child in sorted(method.output_to_method, key=lambda x: x.method_signature):
                    if method.method_path == child.method_path and child.crud < method.crud:
                        # avoid empty path
                        if len(path) > 0:
                            paths.append(path.duplicate())
                        continue
                    if len(paths) > self.threshold_length and len(path) > 0:
                        continue
                    path.append(method)
                    covered_apis.add(method)
                    traverse_path_recursive(child, path)
                    path.remove(method)

        traverse_path_recursive(method, Path())
        return paths, covered_apis

    # FIXME: should remove, only for debugging
    def print_path(self, methods):
        method_signatures = []
        for method in methods:
            method_signatures.append(f'{method.method_type} {method.method_path}')
        res = ' -> '.join(method_signatures)
        print(res)
        return res

    def extend_sequence(self, path):

        feed_from_method = None
        current_method = path[0]
        sequence = Sequence([])
        for i, method in enumerate(path):
            required_property_dict = {}
            if i != 0:
                feed_from_method = path[i - 1]
                current_method = method
                assert feed_from_method != None
                # check for crud and traffic dependency
                required_property_dict = current_method.required_feed_parameter.get(feed_from_method.method_name,
                                                                                    {})
            sequence.add_method(current_method)
            # we do not add reference for the first one
            if i == 0:
                continue

            # add dependency in the yaml (forward analysis)
            for request_parameter in required_property_dict.keys():
                nominal_request_parameters = current_method.get_request_paramter_by_property_name(
                    [request_parameter])
                nominal_feed_response_parameters = feed_from_method.get_response_paramter_by_property_name(
                    [required_property_dict[request_parameter]])
                for request_parameter in nominal_request_parameters:
                    for response_parameter in nominal_feed_response_parameters:
                        sequence.add_def(i - 1, response_parameter)
                        sequence.add_ref(i, feed_from_method, response_parameter,
                                         request_parameter)

            # add dependency from traffic
            if current_method.dependency_from_traffic.__contains__(feed_from_method):
                for dependency in current_method.dependency_from_traffic[feed_from_method]:
                    request_array, response_array = dependency
                    for request_parameter in request_array:
                        for response_parameter in response_array:
                            sequence.add_def(i - 1, response_parameter)
                            sequence.add_ref(i, feed_from_method, response_parameter, request_parameter)

        assert len(path) == len(sequence)
        return sequence

    def generate_sequence(self, simple=False, fast=False):
        crud_sort_map = {
            "head": 1,
            "post": 2,
            "get": 3,
            "put": 4,
            "patch": 5,
            "delete": 6,
        }
        raw_sequences = []
        wrapped_sequences = []
        covered_apis = set()
        all_methods = sorted(self.nodes, key=lambda item: item.method_signature)
        for method in all_methods:
            if simple:
                raw_sequences.append([method])
                continue
            if fast and np.random.random() < 0.5:
                continue
            # FIXME: need review
            if len(method.output_to_method) == 0 and len(method.feed_from_method) > 0 and method in covered_apis:
                continue
            if method in covered_apis and method.crud > crud_sort_map['post']:
                continue
            print('traversing path', method)
            paths, covered = self.generate_graph_sequence(method)
            covered_apis = covered_apis.union(covered)
            raw_sequences.extend(paths)
        for seq in raw_sequences:
            self.print_path(seq)
            sequence = self.extend_sequence(seq)
            sequence.origin = SequenceOrigin.ODG
            wrapped_sequences.append(sequence)
        print('hashing')
        result = set(wrapped_sequences)
        print("Generate sequences # : ", len(raw_sequences), ' extending sequences # : ', len(wrapped_sequences),
              ' result ', len(result))
        return result

    def get_single_node_sequence(self):
        result = []
        for method in self.nodes:
            result.append(Sequence([method]))
        return result
