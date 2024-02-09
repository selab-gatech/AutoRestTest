import random
import string
from dataclasses import dataclass
from typing import List, Dict

import requests
import urllib
import json
from specification_parser import SpecificationParser, ItemProperties, ParameterProperties
from choice import RandomizedSelector

@dataclass
class RequestData:
    endpoint_path: str
    http_method: str
    parameters: Dict
    request_body: Dict
    content_type: str

@dataclass
class StatusCode:
    status_code: int
    count: int
    requests: List[RequestData]

class RequestsGenerator:
    def __init__(self, file_path: str, api_url: str):
        self.file_path = file_path
        self.api_url = api_url
        self.successful_query_data: List[RequestData] = [] # list that will store successfuly query_parameters
        self.status_codes: Dict[int: StatusCode] = {} # dictionary to track status code occurrences

    def process_response(self, response, request_data):
        if response is None:
            return

        if response.status_code not in self.status_codes:
            self.status_codes[response.status_code] = StatusCode(
                status_code=response.status_code,
                count=1,
                requests=[request_data]
            )
        else:
            self.status_codes[response.status_code].count += 1
            self.status_codes[response.status_code].requests.append(request_data)

        if response.status_code // 100 == 2:
            self.successful_query_data.append(request_data)

    def attempt_retry(self, response: requests.Response, request_data: RequestData):
        """
        Attempt retrying request with old query parameters
        """
        if response.status_code // 100 == 2:
            return

        retries = 1
        indices = list(range(len(self.successful_query_data)))
        random.shuffle(indices)
        for i in indices:
            if (200 <= response.status_code < 300) or retries > 5:
                break
            old_request = self.successful_query_data[i]
            if old_request.http_method in {"put", "post"}:
                new_request = RequestData(
                    endpoint_path=request_data.endpoint_path,
                    http_method=request_data.http_method,
                    parameters=old_request.request_body, # use old request body as new query parameters to check for producer-consumer dependency
                    request_body=old_request.request_body,
                    content_type=old_request.content_type
                )
                response = self.send_request(new_request)
                self.process_response(response, new_request)
                retries += 1
        return

    def send_request(self, request_data: RequestData) -> requests.Response:
        """
        Send the request to the API.
        """
        endpoint_path = request_data.endpoint_path
        http_method = request_data.http_method
        query_parameters = request_data.parameters
        request_body = request_data.request_body
        content_type = request_data.content_type
        try:
            select_method = getattr(requests, http_method)
            if http_method in {"put", "post"}:
                if content_type == "json":
                    response = select_method(self.api_url + endpoint_path, params=query_parameters, json=request_body)
                else:
                    response = select_method(self.api_url + endpoint_path, params=query_parameters, data=request_body)
            else:
                response = select_method(self.api_url + endpoint_path, params=query_parameters)
        except requests.exceptions.RequestException:
            print("Request failed")
            return None
        return response

    def randomize_integer(self):
        return random.randint(-2**32, 2**32)

    def randomize_float(self):
        return random.uniform(-2**32, 2**32)

    def randomize_boolean(self):
        return random.choice([True, False])

    def randomize_string(self):
        return ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 9999)))

    def randomize_array(self):
        return [random.randint(-9999, 9999) for _ in range(random.randint(1, 9999))]

    def randomize_object(self):
        return {random.choice(string.ascii_letters): random.randint(-9999, 9999) for _ in range(random.randint(1, 9999))}

    def randomize_null(self):
        return None

    def randomize_parameter_value(self):
        """
        Randomly generate values of any type
        """
        generators = [self.randomize_integer,
                      self.randomize_float,
                      self.randomize_boolean,
                      self.randomize_string,
                      self.randomize_array,
                      self.randomize_object,
                      self.randomize_null]
        return random.choice(generators)()

    def randomize_values(self, parameters, request_body):
        # create randomize object here and return after Object.randomize_parameters() and Object.randomize_request_body() is called
        # do randomize parameter selection, then randomize the values for both parameters and request_body
        randomized_selector = RandomizedSelector()
        return randomized_selector.randomize_parameters(parameters), randomized_selector.randomize_request_body(request_body)

    def randomize_parameters(self, parameter_dict) -> Dict[str, ParameterProperties]:
        """
        Randomly select parameters from the dictionary.
        """
        random_selection = {}
        for parameter_name, parameter_properties in parameter_dict.items():
            if parameter_properties.in_value == "path":
                random_selection[parameter_name] = parameter_properties
            elif random.choice([True, False]):
                random_selection[parameter_name] = parameter_properties
        return random_selection

    def process_operation(self, operation_properties):
        """
        Process the operation properties to generate the request.
        """
        endpoint_path = operation_properties.endpoint_path
        http_method = operation_properties.http_method

        request_body = None
        content_type = None
        if operation_properties.request_body:
            for content_type_value, request_body_properties in operation_properties.request_body_properties.items():
                content_type = content_type_value.replace("application/", "")
                request_body = request_body_properties

        query_parameters, request_body = self.randomize_values(operation_properties.parameters, request_body)

        for parameter_name, parameter_properties in query_parameters.items():
            if parameter_properties.in_value == "path":
                endpoint_path = endpoint_path.replace("{" + parameter_name + "}", str(self.randomize_parameter_value()))

        request_data = RequestData(
            endpoint_path=endpoint_path,
            http_method=http_method,
            parameters=query_parameters,
            request_body=request_body,
            content_type=content_type
        )
        response = self.send_request(request_data)
        self.process_response(response, request_data)
        self.attempt_retry(response, request_data)

    def convert_properties(self, object: ItemProperties):
        if object.type == "array":
            num_objects = random.randint(0, 5)
            obj_arr = []
            for _ in range(num_objects):
                obj_arr.append(self.convert_properties(object.items))
            return obj_arr
        elif object.type == "object":
            object_structure = {}
            for key, value in object.properties.items():
                object_structure[key] = self.convert_properties(value)
            return object_structure
        else:
            return self.randomize_parameter_value()

    
    def requests_generate(self):
        """
        Generate the randomized requests based on the specification file.
        """
        print("Generating Request...")
        print()
        specification_parser = SpecificationParser(self.file_path)
        operations = specification_parser.parse_specification()
        for operation_id, operation_properties in operations.items():
            self.process_operation(operation_properties)
        print()
        print("Generated Request!")

#testing code
if __name__ == "__main__":
    request_generator = RequestsGenerator(file_path="../specs/original/oas/genome-nexus.yaml", api_url="http://localhost:50110")
    request_generator.requests_generate()
    #generate histogram using self.status_code_counts
    print(request_generator.status_code_counts)
    #for i in range(10):
    #    print(request_generator.randomize_parameter_value())