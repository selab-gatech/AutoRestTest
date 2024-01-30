from model.api_path import API
from model.rule.matcher import RuleMatcher
from model.constant import METHOD_CONST
from model.operation_dependency_graph import OperationDependencyGraph


def match_method(methods_first, methods_second, odg):
    for method_first in methods_first:
        for method_second in methods_second:
            odg.add_node(method_second)
            res = RuleMatcher.match(method_first, method_second)
            if res:
                odg.add_edge(method_first, method_second,
                             f'Params: {", ".join(list(method_second.required_feed_parameter[method_first.method_signature].values())[:2])}')


def wrap_base_api(base_path, raw_paths={}):
    res = []
    for path in raw_paths:
        res.append(API(base_path + path, raw_paths[path]))
    return res


def parse(specifiction={}, graph_file_name="graph"):
    paths = specifiction.get("paths")
    base_url = specifiction.get('basePath', "")
    apis = wrap_base_api(base_url, paths)
    odg = OperationDependencyGraph()
    for api_first in apis:
        api_first_method = api_first.methods
        for api_second in apis:
            if api_first == api_second:
                continue
            api_second_method = api_second.methods
            match_method(api_first_method, api_second_method, odg)
    # build crud dependency
    crud_sort_map = {
        METHOD_CONST["head"]: 1,
        METHOD_CONST["post"]: 2,
        METHOD_CONST["get"]: 3,
        METHOD_CONST["put"]: 4,
        METHOD_CONST["patch"]: 5,
        METHOD_CONST["delete"]: 6,
    }
    for api in apis:
        if not len(api.methods) > 1:
            continue
        crud_list = []
        for method in api.methods:
            crud_list.append((crud_sort_map[method.method_type], method))
        crud_list = sorted(crud_list)
        for i in range(1, len(crud_list)):
            to_node = crud_list[i][1]
            method = crud_list[i - 1][1]
            if method.output_to_method.__contains__(to_node):
                continue
            odg.add_edge(method, to_node, 'CRUD')
            method.output_to_method.add(to_node)
            to_node.feed_from_method.add(method)
    for api in apis:
        for method in api.methods:
            odg.add_node(method)
    # draw graph here
    # odg.draw()
    return apis, odg


def parse_concerete(specifiction={}):
    pass
