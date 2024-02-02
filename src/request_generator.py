import random
import string
import requests
from src.specification_parser import SpecificationParser

class RequestsGenerator:

    def __init__(self, file_path: str, api_url: str):
        self.file_path = file_path
        self.api_url = api_url

    def send_request(self, endpoint_path, http_method, query_parameters, request_body_params):
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

        return response

    def randomize_integer(self):
        return random.randint(-9999, 9999)

    def randomize_float(self):
        return random.uniform(-9e99, 9e99)

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
        # care: we allow for 0 parameters to be selected; check if this is okay
        return random_selection

    def process_operation(self, operation_properties):
        """
        Process the operation properties to generate the request.
        """
        endpoint_path = operation_properties.endpoint_path
        http_method = operation_properties.http_method
        selected_parameters = self.randomize_parameters(operation_properties.parameters)

        if operation_properties.request_body:
            # WIP process req body
            pass

        query_parameters = []
        for parameter_name, parameter_values in selected_parameters:
            randomized_value = self.randomize_parameter_value()
            if parameter_values.get("in_value") == "path":
                endpoint_path = endpoint_path.replace("{" + parameter_name + "}", str(randomized_value))
            else:
                query_parameters.append({
                    "parameter_name": parameter_name,
                    "parameter_value": randomized_value
                })

    def requests_generate(self):
        """
        Generate the randomized requests based on the specification file.
        """
        specification_parser = SpecificationParser(self.file_path)
        operations = specification_parser.parse_specification()
        for operation_id, operation_properties in operations.items():
            self.process_operation(operation_properties)

if __name__ == "__main__":
    request_generator = RequestsGenerator(file_path="../specs/original/oas/spotify.yaml")
    request_generator.requests_generate()
    #for i in range(10):
    #    print(request_generator.randomize_parameter_value())