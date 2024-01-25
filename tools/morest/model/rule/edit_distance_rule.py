import editdistance
from model.method import Method
from model.rule.blacklist import BLACK_LIST
from model.util.type_reasoner import reason_type

THRESHOLD = 0.2


class EditDistanceRule:
    @staticmethod
    def match(from_method: Method, to_method: Method):
        response_parameter, request_parameter = from_method.response_parameter_name, to_method.request_parameter_name
        response_parameter_body_tuple = from_method.response_parameter_body_tuple
        request_parameter_body_tuple = to_method.request_parameter_body_tuple
        for response in response_parameter:
            for request in request_parameter:
                if response in BLACK_LIST or request in BLACK_LIST:
                    continue
                if float(editdistance.distance(response, request)) / max(len(request), len(response)) < THRESHOLD:
                    response_list = response_parameter_body_tuple[response]
                    request_list = request_parameter_body_tuple[request]
                    for resp in response_list:
                        for req in request_list:
                            if reason_type(resp[1], req[1]):
                                return True
        return False

    @staticmethod
    def get_params(from_method: Method, to_method: Method):
        parameter_map = {}
        response_parameter, request_parameter = from_method.response_parameter_name, to_method.request_parameter_name
        response_parameter_body_tuple = from_method.response_parameter_body_tuple
        request_parameter_body_tuple = to_method.request_parameter_body_tuple
        for response in response_parameter:
            for request in request_parameter:
                if response in BLACK_LIST or request in BLACK_LIST:
                    continue
                if float(editdistance.distance(response, request)) / max(len(request), len(response)) < THRESHOLD:
                    response_list = response_parameter_body_tuple[response]
                    request_list = request_parameter_body_tuple[request]
                    for resp in response_list:
                        for req in request_list:
                            if reason_type(resp[1], req[1]):
                                parameter_map[request] = response
        return parameter_map
