import argparse
import os

from src.generate_graph import OperationGraph
from src.request_generator import NaiveRequestGenerator

from dotenv import load_dotenv
load_dotenv()

def configure_api_urls(local_test):
    base_port = 9000 if local_test else 8000
    base_url = "0.0.0.0" if local_test else os.getenv("EC2_ADDRESS")
    apis = [
        'fdic', 'genome-nexus', 'language-tool', 'ocvn',
        'ohsome', 'omdb', 'rest-countries', 'spotify', 'youtube'
    ]
    api_urls = {}
    for i, service in enumerate(apis, start=1):
        port = base_port + i
        api_urls[service] = f"http://{base_url}:{port}"
    return api_urls

def parse_args():
    parser = argparse.ArgumentParser(description='Generate requests based on API specification.')
    parser.add_argument("num_specs", choices=["one", "many"],
                        help="Specifies the number of specifications: 'one' or 'many'")
    parser.add_argument("local_test", type=lambda x: (str(x).lower() == 'true'),
                        help="Specifies whether the test is local (true/false)")
    parser.add_argument("-s", "--spec_name", type=str, default=None, help="Optional name of the specification")
    return parser.parse_args()

class AutoRestTest:
    def __init__(self, spec_dir: str, local_test: bool):
        self.spec_dir = spec_dir
        self.local_test = local_test
        self.api_urls = configure_api_urls(local_test)

    def generate_graph(self, spec_name: str):
        spec_path = f"{self.spec_dir}/{spec_name}.yaml"
        operation_graph = OperationGraph(spec_path=spec_path, spec_name=spec_name)
        operation_graph.create_graph()
        return operation_graph

    def generate_requests(self, operation_graph: OperationGraph, api_url: str):
        request_generator = NaiveRequestGenerator(operation_graph, api_url)
        request_generator.generate_requests()

    def run_all(self):
        for spec in os.listdir(self.spec_dir):
            spec_name = spec.split(".")[0]
            print(f"Running tests for {spec_name}")
            operation_graph = self.generate_graph(spec_name)
            api_url = self.api_urls[spec_name]
            self.generate_requests(operation_graph, api_url)

    def run_single(self, spec_name: str):
        operation_graph = self.generate_graph(spec_name)
        api_url = self.api_urls[spec_name]
        self.generate_requests(operation_graph, api_url)

if __name__ == "__main__":
    # example of running: python3 AutoRestTest.py one true -s genome-nexus
    spec_dir = "specs/original/oas"
    args = parse_args()
    auto_rest_test = AutoRestTest(spec_dir, args.local_test)
    if args.num_specs == "one":
        auto_rest_test.run_single(args.spec_name)
    else:
        auto_rest_test.run_all()