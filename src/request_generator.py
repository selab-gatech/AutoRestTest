import random
import string
from dataclasses import dataclass

import requests
import urllib
import json
from specification_parser import SpecificationParser, ItemProperties

@dataclass
class RequestData:
    endpoint_path: str
    http_method: str
    parameters: dict
    request_body: dict

class RequestsGenerator:
    def __init__(self, file_path: str, api_url: str):
        self.file_path = file_path
        self.api_url = api_url
        self.successful_query_data = [] # list that will store successfuly query_parameters
        self.status_code_counts = {} # dictionary to track status code occurrences

    def process_response(self, response, endpoint_path, http_method, query_parameters, request_body=None):
        # Increment the count for the received status code
        self.status_code_counts[response.status_code] = self.status_code_counts.get(response.status_code, 0) + 1
        if response.status_code // 100 == 2:
            self.successful_query_data.append(RequestData(
                endpoint_path=endpoint_path,
                http_method=http_method,
                parameters=query_parameters,
                request_body=request_body
            ))

    def send_request(self, endpoint_path, http_method, query_parameters, request_body=None):
        """
        Send the request to the API.
        """
        try:
            method = getattr(requests, http_method)
            if http_method in {"put", "post"}:
                response = method(self.api_url + endpoint_path, params=query_parameters, json=request_body)
            else:
                response = method(self.api_url + endpoint_path, params=query_parameters)
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

    def randomize_parameters(self, parameter_dict) -> list:
        """
        Randomly select parameters from the dictionary.
        """
        parameter_list = list(parameter_dict.items())
        random_selection = random.sample(parameter_list, k=random.randint(0, len(parameter_list)))
        # careful: we allow for 0 parameters to be selected; check if this is okay
        return random_selection
    
    def process_operation(self, operation_properties):
        """
        Process the operation properties to generate the request.
        """
        endpoint_path = operation_properties.endpoint_path
        http_method = operation_properties.http_method
        selected_parameters = self.randomize_parameters(operation_properties.parameters)


        if operation_properties.request_body:
                parsed_request_body = operation_properties.request_body_properties
                #request_body = self.convert_request_body(parsed_request_body)
                #two cases: parsed_request_body has structure: {MIMETYPE: ItemProperties} or 
                #structure: {MIMETYPE: {KEY: ITEMPROPERTIES}}
                #either way you need to resolve ITEMPROPERTIES based on if it is an item or an array of items, or some other sturcture
                unpacked_request_body = self.convert_request_body(parsed_request_body)
                print(unpacked_request_body)

        query_parameters = []

        for parameter_name, parameter_values in selected_parameters:
            randomized_value = self.randomize_parameter_value()
            if parameter_values.in_value == "path":
                endpoint_path = endpoint_path.replace("{" + parameter_name + "}", str(randomized_value))
            else:
                query_parameters[parameter_name] = randomized_value

        #making request and storing return value
        response = self.send_request(endpoint_path, http_method, query_parameters, request_body)

        if response is not None:
            #processing the response if request was successful
            self.process_response(response, endpoint_path, http_method, query_parameters, request_body)

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
    
    def convert_request_body(self, parsed_request_body):
        if 'application/json' in parsed_request_body:
            object = parsed_request_body['application/json']
            if isinstance(object, ItemProperties):
                constructed_body = self.convert_properties(object)
                return json.dumps(constructed_body)
            elif isinstance(object, list):
                arr = []
                for obj in object:
                    arr.append(self.convert_properties(obj))
                return json.dumps(arr)
            else:
                raise SyntaxError("Request Body Schema Parsing Error")
        elif 'application/x-www-form-urlencoded' in parsed_request_body:
            object = parsed_request_body['application/x-www-form-urlencoded']
            if isinstance(object, ItemProperties):
                constructed_body = self.convert_properties(object)
                return urllib.urlencode(constructed_body)
            elif isinstance(object, list):
                arr = []
                for obj in object:
                    arr.append(self.convert_properties(obj))
                return urllib.urlencode(arr)
            else:
                raise SyntaxError("Request Body Schema Parsing Error")
        else:
          keys = list(parsed_request_body.keys())
          if len(keys) == 1:
            raise ValueError("Unsupported MIME type: " + keys[0] + " in Request Body Specification")
          else:
              raise SyntaxError("Formatting Error in Specification")
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