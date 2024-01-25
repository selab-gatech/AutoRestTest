from model.method import Method
from model.constant import METHOD_CONST
from model.rule.matcher import RuleMatcher


class API:
    def __init__(self, path, body={}):
        self.path = path
        self.raw_body = body
        self.methods = []
        self.method_map = {}
        self.response_parameters = set()
        self.request_parameters = set()
        self.parse_body(self.raw_body)
        self.crud_sematic()

    def parse_body(self, body):
        methods = []
        for method in body:
            # work around for bitbucket
            if method == 'parameters':
                continue
            if body.__contains__('parameters'):
                if not body[method].__contains__('parameters'):
                    body[method]['parameters'] = []
                body[method]['parameters'].extend(body['parameters'])
            if len(body[method].keys()) == 0:
                continue
            print(method, self.path)
            method = self.wrap_method(method, self.path, body[method])
            methods.append(method)
            self.method_map[method] = method
        for method in methods:
            self.request_parameters = set.union(self.request_parameters, method.request_parameter_name)
            self.response_parameters = set.union(self.response_parameters, method.response_parameter_name)
        self.methods = methods

    def wrap_method(self, method, path, method_body={}):
        return Method(method, path, method_body)

    def crud_sematic(self):
        post_method = None
        if self.method_map.__contains__("post"):
            post_method = self.method_map["post"]
        for from_method in self.methods:
            for to_method in self.methods:
                if from_method == to_method or to_method == post_method:
                    continue
                RuleMatcher.match(from_method, to_method)
        if post_method != None:
            for method in self.methods:
                if post_method == method:
                    continue
                method.feed_from_method.add(post_method)

    def __str__(self):
        print(self.path)
