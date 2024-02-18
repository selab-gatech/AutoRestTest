import random
import string
from dataclasses import dataclass
from typing import List, Dict
import argparse
import threading
import requests
import urllib
import json
from specification_parser import SpecificationParser, ItemProperties, ParameterProperties, OperationProperties
from randomizer import RandomizedSelector


@dataclass
class RequestData:
    endpoint_path: str
    http_method: str
    parameters: Dict[str, any]
    request_body: any
    content_type: str
    operation_id: str

@dataclass
class StatusCode:
    status_code: int
    count: int
    requests: List[RequestData]

class RequestsGenerator:
    def __init__(self, file_path: str, api_url: str, is_local: bool = False):
        self.file_path = file_path
        self.api_url = api_url
        self.successful_query_data: List[RequestData] = [] # list that will store successfuly query_parameters
        self.status_codes: Dict[int: StatusCode] = {} # dictionary to track status code occurrences
        self.specification_parser: SpecificationParser = SpecificationParser(self.file_path)
        self.operations: Dict[str, OperationProperties] = self.specification_parser.parse_specification()
        self.is_local = is_local

    def get_simple_type(self, variable):
        """
        Returns a simplified type name as a string for common Python data types.
        """
        type_mapping = {
            int: "integer",
            float: "float",
            str: "string",
            list: "array",
            dict: "object",
            tuple: "tuple",
            set: "set",
            bool: "boolean",
            type(None): "null"
        }
        var_type = type(variable)
        return type_mapping.get(var_type, str(var_type))

    def determine_composite_items(self, item_properties: ItemProperties, curr_value) -> ItemProperties:
        if item_properties.type == "array" and len(curr_value) > 0:
            item_properties.items = ItemProperties(
                type=self.get_simple_type(curr_value[0])
            )
        elif item_properties.type == "object":
            item_properties.properties = {}
            for key, value in curr_value.items():
                item_properties.properties[key] = ItemProperties(
                    type=self.get_simple_type(value)
                )
        return item_properties

    def create_operation_for_mutation(self, query_value: RequestData, operation_properties: OperationProperties) -> OperationProperties:
        """
        Create a new operation for mutation
        """
        if query_value.request_body and operation_properties.request_body_properties:
            for content_type, request_body_properties in operation_properties.request_body_properties.items():
                request_body_properties = ItemProperties(
                    type=self.get_simple_type(query_value.request_body)
                )
                request_body_properties = self.determine_composite_items(request_body_properties, query_value.request_body)
                operation_properties.request_body_properties[content_type] = request_body_properties

        endpoint_path = operation_properties.endpoint_path
        for parameter_name, parameter_properties in operation_properties.parameters.items():
            if parameter_properties.in_value == "path":
                manual_randomizer = RandomizedSelector(operation_properties.parameters, query_value.request_body)
                operation_properties.endpoint_path = endpoint_path.replace(
                    "{" + parameter_name + "}", str(manual_randomizer.randomize_item(parameter_properties.schema)))

        operation_properties.parameters = {}
        if query_value.parameters:
            for parameter_name, parameter_value in query_value.parameters.items():
                parameter_properties = ParameterProperties(
                    name=parameter_name,
                    in_value="query",
                    schema=ItemProperties(
                        type=self.get_simple_type(parameter_value)
                    )
                )
                parameter_properties.schema = self.determine_composite_items(parameter_properties.schema, parameter_value)
                operation_properties.parameters[parameter_name] = parameter_properties

        return operation_properties

    def mutate_requests(self):
        """
        Mutate valid queries for further testing
        """
        print("Mutating Requests...")
        curr_success_queries = self.successful_query_data.copy()
        for query in curr_success_queries:
            curr_id = query.operation_id
            operation_details = self.operations.get(curr_id)
            if operation_details is not None:
                new_operation: OperationProperties = operation_details
                new_operation = self.create_operation_for_mutation(query, new_operation)
                self.process_operation(new_operation)

    def process_response(self, response, request_data):
        """
        Process the response from the API.
        """
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
        if response is None or response.status_code // 100 == 2:
            return

        retries = 1
        indices = list(range(len(self.successful_query_data)))
        random.shuffle(indices)
        for i in indices:
            if response is None or (200 <= response.status_code < 300) or retries > 5:
                break
            old_request = self.successful_query_data[i]
            if old_request.http_method in {"put", "post"}:
                new_request = RequestData(
                    endpoint_path=request_data.endpoint_path,
                    http_method=request_data.http_method,
                    parameters=old_request.request_body, # use old request body as new query parameters to check for producer-consumer dependency
                    request_body=old_request.request_body,
                    content_type=old_request.content_type,
                    operation_id=request_data.operation_id
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
                    response = select_method(self.api_url + endpoint_path, params=query_parameters, json=json.dumps(request_body))
                else:
                    response = select_method(self.api_url + endpoint_path, params=query_parameters, data=request_body)
            else:
                response = select_method(self.api_url + endpoint_path, params=query_parameters)
        except requests.exceptions.RequestException as e:
            print("Request failed")
            return None
        return response

    def randomize_values(self, parameters: Dict[str, ParameterProperties], request_body) -> (Dict[str, any], any):
        # create randomize object here and return after Object.randomize_parameters() and Object.randomize_request_body() is called
        # do randomize parameter selection, then randomize the values for both parameters and request_body
        randomizer = RandomizedSelector(parameters, request_body)
        return randomizer.randomize_parameters() if parameters else None, randomizer.randomize_request_body() if request_body else None

    def process_operation(self, operation_properties: OperationProperties):
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

        for parameter_name, parameter_properties in operation_properties.parameters.items():
            if parameter_properties.in_value == "path":
                manual_randomizer = RandomizedSelector(operation_properties.parameters, request_body)
                endpoint_path = endpoint_path.replace("{" + parameter_name + "}", str(manual_randomizer.randomize_item(parameter_properties.schema)))

        request_data = RequestData(
            endpoint_path=endpoint_path,
            http_method=http_method,
            parameters=query_parameters,
            request_body=request_body,
            content_type=content_type,
            operation_id=operation_properties.operation_id
        )
        print("Request Sent")
        response = self.send_request(request_data)
        if response is not None:
            self.process_response(response, request_data)
            self.attempt_retry(response, request_data)

    def requests_generate(self):
        """
        Generate the randomized requests based on the specification file.
        """
        print("Generating Request...")
        print()
        if not self.is_local:
            num_workers = 5
            worker_queues = [[] for i in range(num_workers)] 
            for i, (operation_id, operation_properties) in enumerate(self.operations.items()):
                worker_queues[i % num_workers].append((operation_id, operation_properties))
            workers = []
            for i in range(num_workers):
                worker = threading.Thread(target=self.process_operation, args=(worker_queues[i],))
                workers.append(worker)
                worker.start()
            for worker in workers:
                worker.join()
        else:
            for operation_id, operation_properties in self.operations.items():
                self.process_operation(operation_properties)

        self.mutate_requests()
        print("Generated Requests!")

service_urls = {
    'fdic': "http://0.0.0.0:9001",
    'genome-nexus': "http://0.0.0.0:9002",
    'language-tool': "http://0.0.0.0:9003",
    'ocvn': "http://0.0.0.0:9004",
    'ohsome': "http://0.0.0.0:9005",
    'omdb': "http://0.0.0.0:9006",
    'rest-countries': "http://0.0.0.0:9007",
    'spotify': "http://0.0.0.0:9008",
    'youtube': "http://0.0.0.0:9009"
}
#testing code
if __name__ == "__main__":
    # Set up argparse to handle command line arguments
    parser = argparse.ArgumentParser(description='Generate requests based on API specification.')
    parser.add_argument('service', help='The service specification to use.')
    # Parse the command line arguments
    args = parser.parse_args()
    # Get the api_url from the dictionary using the service name provided
    api_url = service_urls.get(args.service)
    if api_url is None:
        print(f"Service '{args.service}' not recognized. Available services are: {list(service_urls.keys())}")
        exit(1)
    file_path = f"../specs/original/oas/{args.service}.yaml"
    request_generator = RequestsGenerator(file_path=file_path, api_url=api_url, is_local=True)
    for i in range(5):
        request_generator.requests_generate()
        print(i)
    #generate histogram using self.status_code_counts
    print([(x.status_code, x.count) for x in request_generator.status_codes.values()])
    #for i in range(10):
    #    print(request_generator.randomize_parameter_value())