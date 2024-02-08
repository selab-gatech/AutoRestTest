import random
import string
import requests
import urllib
import json
from specification_parser import SpecificationParser, ItemProperties


class RequestsGenerator:
    def __init__(self, file_path: str, api_url: str):
        self.file_path = file_path
        self.api_url = api_url
        self.successful_query_parameters = [] #list that will store successfuly query_parameters
        self.status_code_counts = {} #dictionary to track status code occurrences

    def process_response(self, response, query_parameters):
    
        if response is not None:
        # Increment the count for the received status code
            if response.status_code in self.status_code_counts:
                self.status_code_counts[response.status_code] += 1
            else:
                self.status_code_counts[response.status_code] = 1

            if response.status_code >= 200 and response.status_code < 300:
                self.successful_query_parameters.append(query_parameters)
    
    def send_request(self, endpoint_path, http_method, query_parameters):
        """
        Send the request to the API.
        """
        if http_method == "get":
            try:
                response = requests.get(self.api_url + endpoint_path, params=query_parameters)
            except requests.exceptions.RequestException as e:
                return None
        elif http_method == "post":
            try:
                response = requests.post(self.api_url + endpoint_path, params=query_parameters)
            except requests.exceptions.RequestException as e:
                return None
        elif http_method == "put":
            try:
                response = requests.put(self.api_url + endpoint_path, params=query_parameters)
            except requests.exceptions.RequestException as e:
                return None
        elif http_method == "delete":
            try:
                response = requests.delete(self.api_url + endpoint_path, params=query_parameters)
            except requests.exceptions.RequestException as e:
                return None
        else:
            raise ValueError("Invalid HTTP method")

        return response, query_parameters

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
    
    def randomize_parameters(self, parameter_dict):
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
                print("__BEGIN REQUEST BODY__")
                print()
                print(parsed_request_body)
                print()
                print("___END REQUEST BODY__")
                #request_body = self.convert_request_body(parsed_request_body)
                #two cases: parsed_request_body has structure: {MIMETYPE: ItemProperties} or 
                #structure: {MIMETYPE: {KEY: ITEMPROPERTIES}}
                #either way you need to resolve ITEMPROPERTIES based on if it is an item or an array of items, or some other sturcture
                
        query_parameters = []
        for parameter_name, parameter_values in selected_parameters:
            randomized_value = self.randomize_parameter_value()
            if parameter_values.in_value == "path":
                endpoint_path = endpoint_path.replace("{" + parameter_name + "}", str(randomized_value))
            else:
                query_parameters.append({
                    "parameter_name": parameter_name,
                    "parameter_value": randomized_value
                })
        
        #converting query_parameters to a dictionary because send_request expects a dict
        # query_parameters_dict = {}
        # for parameter_name, parameter_values in selected_parameters:
        #     randomized_value = self.randomize_parameter_value()
        #     query_parameters_dict[parameter_name] = randomized_value
        
        #making request and storing return value in variables
        response, used_query_parameters = self.send_request(endpoint_path, http_method, query_parameters)
        #processing the response
        self.process_response(response, used_query_parameters)

    def convert_properties(self, object: ItemProperties):
        if object.type == "array":
            pass
        else:
            return 
    
    def convert_request_body(self, parsed_request_body):
        if 'application/json' in parsed_request_body:
            # json_body = {key: self.convert_properties(properties) for key, properties in parsed_request_body['application/json'].items()}
            # return json_body
            object = parsed_request_body['application/json']
            if isinstance(object, ItemProperties):
                constructed_body = self.convert_properties(object)
                return json.dumps(constructed_body)
            elif isinstance(object, list):
                pass 
            elif isinstance(object, dict):
                pass
            else:
                raise SyntaxError("Request Body Schema Parsing Error")
        elif 'application/x-www-form-urlencoded' in parsed_request_body:
            form_data = {key: self.convert_properties(properties) for key, properties in parsed_request_body['application/x-www-form-urlencoded'].items()}
            return urllib.urlencode(form_data)
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