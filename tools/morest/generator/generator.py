import copy
from model.method import Method
import json
import random
import string


def randomString(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def traverse_root(method):
    res = []

    def recusive_traverse(root, path=[]):
        path.append(root)
        res.append(path)
        for output in root.output_to_method:
            recusive_traverse(output, copy.deepcopy(path))

    recusive_traverse(method, [])
    return res


def generate_testcases(testcases=[]):
    res = {}
    ind = 0
    type = set()

    def assign_value(parameter={}):
        should_change = False
        for key in parameter.keys():
            if key == "type":
                should_change = True
            if isinstance(parameter[key], dict):
                assign_value(parameter[key])
        if should_change:
            try:
                type.add(parameter["type"])
                parameter["value"] = "value"
            except:
                pass

    def extract_parameters(parameters={}):
        res = {}
        for name in parameters.keys():
            res[name] = copy.deepcopy(parameters[name].raw_body)
            assign_value(res[name])
        return res

    def extract_steps(methods=[]):
        steps = []
        for method in methods:
            step = {}
            step["type"] = method.method_type
            step["path"] = method.method_path
            step["request_parameters"] = extract_parameters(method.request_parameters)
            step["required_parameters"] = list(method.required_feed_parameter)
            steps.append(step)
        return steps

    for testcase in testcases:
        ind += 1
        name = f'testcase_{ind}'
        json_testcase = {}
        json_testcase["steps"] = extract_steps(testcase)
        res[name] = json_testcase

    with open("testcase.json", 'w') as data:
        json.dump(res, data)
    print(type)


def traverser(apis=[]):
    testcases = []
    for api in apis:
        for method in api.methods:
            if len(method.feed_from_method) != 0:
                continue
            cases = traverse_root(method)
            for case in cases:
                testcases.append(case)
    generate_testcases(testcases)
