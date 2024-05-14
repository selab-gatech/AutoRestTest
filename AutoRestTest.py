import argparse
import os

from src.generate_graph import OperationGraph
from src.request_generator import RequestGenerator

from dotenv import load_dotenv

from src.specification_parser import SpecificationParser

load_dotenv()

def get_api_url(spec_parser: SpecificationParser, local_test: bool):
    api_url = spec_parser.get_api_url()
    if not local_test:
        api_url = api_url.replace("localhost", os.getenv("EC2_ADDRESS"))
        api_url = api_url.replace(":9", ":8")
    return api_url

def parse_args():
    parser = argparse.ArgumentParser(description='Generate requests based on API specification.')
    parser.add_argument("num_specs", choices=["one", "many"],
                        help="Specifies the number of specifications: 'one' or 'many'")
    parser.add_argument("local_test", type=lambda x: (str(x).lower() == 'true'),
                        help="Specifies whether the test is local (true/false)")
    parser.add_argument("-s", "--spec_name", type=str, default=None, help="Optional name of the specification")
    return parser.parse_args()

class AutoRestTest:
    def __init__(self, spec_dir: str, local_test: bool, is_naive=False):
        self.spec_dir = spec_dir
        self.local_test = local_test
        self.is_naive = is_naive

    def init_graph(self, spec_name: str, spec_path) -> OperationGraph:
        spec_parser = SpecificationParser(spec_path=spec_path, spec_name=spec_name)
        api_url = get_api_url(spec_parser, self.local_test)
        operation_graph = OperationGraph(spec_path=spec_path, spec_name=spec_name, spec_parser=spec_parser)
        request_generator = RequestGenerator(operation_graph=operation_graph, api_url=api_url, is_naive=self.is_naive)
        operation_graph.assign_request_generator(request_generator)
        return operation_graph

    def generate_graph(self, spec_name: str):
        spec_path = f"{self.spec_dir}/{spec_name}.yaml"
        operation_graph = self.init_graph(spec_name, spec_path)
        operation_graph.create_graph()
        return operation_graph

    def run_all(self):
        for spec in os.listdir(self.spec_dir):
            spec_name = spec.split(".")[0]
            print(f"Running tests for {spec_name}")
            self.run_single(spec_name)

    def run_single(self, spec_name: str):
        operation_graph = self.generate_graph(spec_name)

if __name__ == "__main__":
    # example of running: python3 AutoRestTest.py one true -s genome-nexus
    spec_dir = "specs/original/oas"
    args = parse_args()
    auto_rest_test = AutoRestTest(spec_dir=spec_dir, local_test=args.local_test, is_naive=False)
    if args.num_specs == "one":
        auto_rest_test.run_single(args.spec_name)
    else:
        auto_rest_test.run_all()