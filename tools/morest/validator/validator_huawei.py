import json

from flex.core import validate as flex_validate


class ViolationType:
    SCHEMA_VIOLATION = "SCHEMA_VIOLATION"
    STATUS_CODE = "STATUS_CODE"
    SCHEMA_MISSING = "SCHEMA_MISSING"


class Violation:
    def __init__(self, method, api_response, violate_type):
        self.method = method
        self.method_name = method.method_name
        self.method_path = method.method_path
        self.api_response = api_response
        self.violate_type = violate_type
        self.schema_violations = None

    def __str__(self):
        str_result = f'Request ID:{self.api_response.request_id}, Method: {self.method_name}, Method Path: {self.method_path},' \
                     f' Status Code:{self.api_response.status_code}, Violation:{self.violate_type} ,'
        if self.violate_type == ViolationType.SCHEMA_VIOLATION:
            str_result += f'Violation Messages: {self.schema_violations} ,'
        str_result += f'Request Content: {self.api_response.content}'
        return str_result

    def signature(self):
        return f'{self.api_response.api_name}-{self.violate_type}-{self.api_response.status_code}'

    def json(self):
        result = {
            "requestID": self.api_response.request_id,
            "method": self.method_name,
            "methodPath": self.method_path,
            "statusCode": self.api_response.status_code,
            "violation": self.violate_type,
            "requestContent": self.api_response.content
        }
        if self.violate_type == ViolationType.SCHEMA_VIOLATION:
            result["violationMessage"] = self.schema_violations
        return result


class Reponse:
    def __init__(self, body):
        self.request_id = body["resquestId"]
        self.method = body["method"]
        self.api_name = body["apiName"]
        self.url = body["url"]
        self.status_code = body["statusCode"]
        self.content = None
        if body.__contains__("content"):
            try:
                self.content = json.loads(body["content"])
            except:
                self.content = body["content"]
        self.response = body


def wrap_response(response):
    return Reponse(response)


def resolve_response(data):
    responses = data
    result = []
    for resp in responses:
        result.append(wrap_response(resp))
    return result


def find_method_by_response(apis, response):
    api_name = response.api_name
    for api in apis:
        for method in api.methods:
            if method.method_name == api_name:
                return method
    print(api_name, "not found ")
    assert False


def validate_nominal(method, response):
    results = []
    # FIXME: bypass status 0 in huawei
    if response.status_code == 0:
        return results
    if response.status_code >= 500:
        results.append(Violation(method, response, ViolationType.STATUS_CODE))
    if not method.response_parameter.__contains__(str(response.status_code)):
        results.append(Violation(method, response, ViolationType.SCHEMA_MISSING))
    if method.response_parameter.__contains__(str(response.status_code)):
        schema = method.response_parameter[str(response.status_code)].schema
        try:
            flex_validate(schema, response.content)
        except Exception as ex:
            if schema != None:
                violation = Violation(method, response, ViolationType.SCHEMA_VIOLATION)
                violation.schema_violations = str(ex)
                results.append(violation)
    return results


def validate_non_nominal(method, response):
    results = []
    # FIXME: bypass status 0 in huawei
    if response.status_code == 0:
        return results
    if response.status_code >= 500 and not method.response_parameter.__contains__(str(response.status_code)):
        results.append(Violation(method, response, ViolationType.STATUS_CODE))
    if not method.response_parameter.__contains__(str(response.status_code)):
        results.append(Violation(method, response, ViolationType.SCHEMA_MISSING))
    if method.response_parameter.__contains__(str(response.status_code)):
        schema = method.response_parameter[str(response.status_code)].schema
        try:
            flex_validate(schema, response.content)
        except Exception as ex:
            violation = Violation(method.method_name, response, ViolationType.SCHEMA_VIOLATION)
            violation.schema_violations = str(ex)


def validate(resp_data, apis, nominal=True):
    responses = resolve_response(resp_data)
    result = []
    for response in responses:
        method = find_method_by_response(apis, response)
        if nominal:
            result.extend(validate_nominal(method, response))
        else:
            result.extend(validate_non_nominal(method, response))
    return result
