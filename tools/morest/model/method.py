import uuid
import re
from model.constant import METHOD_CONST
from model.parameter import Parameter

crud_sort_map = {
    "head": 1,
    "post": 2,
    "get": 3,
    "put": 4,
    "patch": 5,
    "delete": 6,
}


class Method:
    def __init__(self, method, path, method_body={}, alternative_method=None):
        self.method_type = METHOD_CONST[method]
        self.method_path = path
        self.method_mapping_path = path
        self.method_body = method_body
        self.method_id = uuid.uuid4().__str__()
        self.method_name = method_body.get("operationId", f'{self.method_type}-{self.method_path}')
        self.request_parameters = {}
        self.request_parameter_name = set()
        self.response_parameter = {}
        self.response_parameter_name = set()
        self.request_parameter_body_tuple = {}
        self.response_parameter_body_tuple = {}
        self.feed_from_method = set()
        self.output_to_method = set()
        self.required_feed_parameter = {}
        self.dependency_from_traffic = {}
        self.method_signature = f'{self.method_type}-{self.method_path}'
        self.crud = crud_sort_map[self.method_type]
        # this is used for alternative method.py mapping
        if alternative_method:
            self.crud = crud_sort_map[alternative_method]
        self.parse(self.method_body)

    def parse(self, body={}):
        # parse request
        if body.__contains__("parameters"):
            req_parameters = body["parameters"]
            req_parameters = self.parse_param(req_parameters)
            self.request_parameters = req_parameters
        # parse response
        responses = body["responses"]
        for respon in responses:
            format_response = self.parse_response(respon, responses[respon])
        print("Response", self.response_parameter_name)

    def parse_param(self, parameters=[]):
        res = {}
        for param in parameters:
            name = param["name"]
            res[name] = Parameter(name, param, self.method_path)
            self.request_parameter_name = set.union(self.request_parameter_name, res[name].parameter_names)
            for k in res[name].parameter_body_tuple.keys():
                body_tuple_list = self.request_parameter_body_tuple.get(k, [])
                body_tuple_list.extend(res[name].parameter_body_tuple[k])
                self.request_parameter_body_tuple[k] = body_tuple_list
        print("Request", self.request_parameter_name)
        return res

    def parse_response(self, status_code, response={}):
        res = {}
        # if self.method_signature == 'get-/rest/blockservice/v1/lun-groups/{id}':
        #     print()
        parameter = Parameter(str(status_code), response, self.method_path)
        self.response_parameter[str(status_code)] = parameter
        self.response_parameter_name = set.union(self.response_parameter_name, parameter.parameter_names)
        self.response_parameter_name.remove(str(status_code))
        for k in parameter.parameter_body_tuple.keys():
            body_tuple_list = self.response_parameter_body_tuple.get(k, [])
            body_tuple_list.extend(parameter.parameter_body_tuple[k])
            self.response_parameter_body_tuple[k] = body_tuple_list
        return res

    def get_parameter_property_name(self, param_property_name=""):
        result, candidate_name = None, None
        for response in self.response_parameter.values():
            if not response.parameter_names.__contains__(param_property_name):
                continue
            candidate_pool = list(response.attribute_path_dict[param_property_name])
            # FIXME: SHOULD BE PROPER HANDLED FOR SPANNING PARAMETER IN DIFFERENT SEQUENCES
            result = response.parameter_id + candidate_pool[0]
            candidate_name = candidate_pool[0]
        assert result
        assert candidate_name
        return result, candidate_name

    def _get_nominal_parameters(self, parameters, with_parameter_name=True):
        res = set()
        if len(parameters) == 0:
            print(self.method_signature, 'has no parameter [in get nominal parameters]')
        for parameter in parameters:
            for nominal_values in parameter.attribute_path_dict.values():
                for value in nominal_values:
                    if with_parameter_name and value != parameter.name:
                        res.add(f'{parameter.name}.{value}')
                    else:
                        res.add(value)
        return res

    def get_nominal_request_parameter(self):
        return self._get_nominal_parameters(self.request_parameters.values())

    def get_request_paramter_by_property_name(self, properties):
        res = self._get_nominal_name_by_property_name(properties, self.request_parameters.values(), True)
        assert len(res) > 0
        return res

    def get_response_paramter_by_property_name(self, properties):
        res = self._get_nominal_name_by_property_name(properties, self.response_parameter.values())
        assert len(res) > 0
        return res

    def get_single_request_parameter_by_property_name(self, prop):
        res = self._get_nominal_name_by_property_name([prop], self.request_parameters.values())
        assert len(res) > 0
        return res

    def get_single_response_parameter_by_property_name(self, prop):
        res = self._get_nominal_name_by_property_name([prop], self.response_parameter.values())
        assert len(res) > 0
        return res

    def get_feed_from_method_response_parameter_by_property_and_method_name(self, method_name, prop):
        for method in self.feed_from_method:
            if method.method_signature == method_name:
                return method.get_single_response_parameter_by_property_name(prop)

    def _get_nominal_name_by_property_name(self, properties, parameters, with_parameter_name=False):
        res = set()
        for parameter in parameters:
            for prop in properties:
                if not parameter.attribute_path_dict.__contains__(prop):
                    continue
                nominal_dict = set()
                for attribute in parameter.attribute_path_dict[prop]:
                    # to avoid add par1.par1 (should add par1 directly)
                    if with_parameter_name and attribute != parameter.name:
                        nominal_dict.add(f'{parameter.name}.{attribute}')
                    else:
                        nominal_dict.add(attribute)

                res = res.union(nominal_dict)
        return res

    def __str__(self):
        return self.method_signature

    def __hash__(self):
        return hash(self.method_signature)
