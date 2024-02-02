import random
import string
from src.specification_parser import SpecificationParser

class RequestsGenerator:

    def __init__(self, file_path: str):
        self.file_path = file_path

    def randomize_parameters(self, parameter_dict):
        """
        Randomly select parameters from the dictionary.
        """
        parameter_list = list(parameter_dict.items())
        random_selection = random.sample(parameter_list, k=random.randint(0, len(parameter_list)))
        # care: we allow for 0 parameters to be selected; check if this is okay
        return random_selection

    def randomize_parameter_value(self):
        """
        Randomly generate values of any type
        """
        generators = [
            lambda: random.randint(-9999, 9999),
            lambda: random.uniform(-9e99, 9e99),
            lambda: random.choice([True, False]),
            lambda: ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 9999))),
            lambda: [random.randint(-9999, 9999) for _ in range(random.randint(1, 9999))],
            lambda: {random.choice(string.ascii_letters): random.randint(-9999, 9999) for _ in range(random.randint(1, 9999))},
            lambda: None
        ]
        return random.choice(generators)()


    def process_operation(self, operation_properties):
        """
        Process the operation properties to generate the request.
        """
        endpoint_path = operation_properties.endpoint_path
        http_method = operation_properties.http_method
        selected_parameters = self.randomize_parameters(operation_properties.parameters)

        query_parameters = []
        for parameter_name, parameter_values in selected_parameters:
            randomized_value = self.randomize_parameter_value()
            query_parameters.append({"parameter_name": parameter_name, "parameter_value": randomized_value})


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