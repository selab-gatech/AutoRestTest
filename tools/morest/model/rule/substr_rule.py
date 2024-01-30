from model.rule.blacklist import BLACK_LIST
from model.util.type_reasoner import reason_type


class SubStringRule:
    @staticmethod
    def match(from_method, to_method):
        response_parameter, request_parameter = from_method.response_parameter_name, to_method.request_parameter_name
        response_parameter_body_tuple = from_method.response_parameter_body_tuple
        request_parameter_body_tuple = to_method.request_parameter_body_tuple
        for response in response_parameter:
            for request in request_parameter:
                if response in BLACK_LIST or request in BLACK_LIST:
                    continue
                if response.lower().find(request.lower()) > -1 or request.lower().find(response.lower()) > -1:
                    response_list = response_parameter_body_tuple[response]
                    request_list = request_parameter_body_tuple[request]
                    for resp in response_list:
                        for req in request_list:
                            if reason_type(resp[1], req[1]):
                                return True
        return False

    # @staticmethod
    # def get_params(from_method, to_method):
    #     parameter_map = {}
    #     response_parameter, request_parameter = from_method.response_parameter_name, to_method.request_parameter_name
    #     matched_params = set.intersection(response_parameter, request_parameter)
    #     response_parameter_body_tuple = from_method.response_parameter_body_tuple
    #     request_parameter_body_tuple = to_method.request_parameter_body_tuple
    #     for param in matched_params:
    #         response_list = response_parameter_body_tuple[param]
    #         request_list = request_parameter_body_tuple[param]
    #         for response in response_list:
    #             for request in request_list:
    #                 if reason_type(response[1], request[1]):
    #                     parameter_map[param] = param
    #     return parameter_map
    #
    # @staticmethod
    # def match(response_parameter=set(), request_parameter=set()):
    #     for p1 in response_parameter:
    #         for p2 in request_parameter:
    #             if (p1.lower().find(p2.lower()) > -1 or p2.lower().find(p1.lower())) > -1 and (
    #             not (p1 in BLACK_LIST)) and (
    #                     not (p2 in BLACK_LIST)):
    #                 return True
    #     return False

    @staticmethod
    def get_params(from_method, to_method):
        res = {}
        response_parameter, request_parameter = from_method.response_parameter_name, to_method.request_parameter_name
        response_parameter_body_tuple = from_method.response_parameter_body_tuple
        request_parameter_body_tuple = to_method.request_parameter_body_tuple
        for response in response_parameter:
            for request in request_parameter:
                if response in BLACK_LIST or request in BLACK_LIST:
                    continue
                if response.lower().find(request.lower()) > -1 or request.lower().find(response.lower()) > -1:
                    response_list = response_parameter_body_tuple[response]
                    request_list = request_parameter_body_tuple[request]
                    for resp in response_list:
                        for req in request_list:
                            if reason_type(resp[1], req[1]):
                                res[request] = response
        return res
