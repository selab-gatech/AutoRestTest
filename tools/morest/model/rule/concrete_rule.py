from model.rule.blacklist import BLACK_LIST
from model.util.type_reasoner import reason_type


class ConcreteRule:
    @staticmethod
    def match(from_method, to_method):
        response_parameter, request_parameter = from_method.response_parameter_name, to_method.request_parameter_name
        if len(set.difference(set.intersection(response_parameter, request_parameter), BLACK_LIST)) == 0:
            return False
        matched_params = set.intersection(response_parameter, request_parameter)
        response_parameter_body_tuple = from_method.response_parameter_body_tuple
        request_parameter_body_tuple = to_method.request_parameter_body_tuple
        for param in matched_params:
            response_list = response_parameter_body_tuple[param]
            request_list = request_parameter_body_tuple[param]
            for response in response_list:
                for request in request_list:
                    if reason_type(response[1], request[1]):
                        return True
        return False

    @staticmethod
    def get_params(from_method, to_method):
        # if from_method.method_signature == 'get-/rest/blockservice/v1/lun-groups/{id}':
        #     print()
        parameter_map = {}
        response_parameter, request_parameter = from_method.response_parameter_name, to_method.request_parameter_name
        matched_params = set.intersection(response_parameter, request_parameter)
        response_parameter_body_tuple = from_method.response_parameter_body_tuple
        request_parameter_body_tuple = to_method.request_parameter_body_tuple
        for param in matched_params:
            response_list = response_parameter_body_tuple[param]
            request_list = request_parameter_body_tuple[param]
            for response in response_list:
                for request in request_list:
                    if reason_type(response[1], request[1]):
                        parameter_map[param] = param
        return parameter_map
