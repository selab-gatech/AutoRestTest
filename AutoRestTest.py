import argparse
import json
import os
import shelve

from src.generate_graph import OperationGraph
from src.marl import QLearning
from src.request_generator import RequestGenerator

from dotenv import load_dotenv

from src.graph.specification_parser import SpecificationParser
from src.utils import OpenAILanguageModel, _construct_db_dir

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

def output_q_table(q_learning, spec_name):
    parameter_table = q_learning.parameter_agent.q_table
    body_obj_table = q_learning.body_object_agent.q_table
    #header_table = q_learning.header_agent.q_table
    value_table = q_learning.value_agent.q_table
    operation_table = q_learning.operation_agent.q_table
    data_source_table = q_learning.data_source_agent.q_table
    dependency_table = q_learning.dependency_agent.q_table

    simplified_param_table = {}
    for operation, operation_values in parameter_table.items():
        simplified_param_table[operation] = {"params": {}, "body": {}}
        for parameter, parameter_values in operation_values["params"].items():
            simplified_param_table[operation]["params"][str(parameter)] = parameter_values
        for body, body_values in operation_values["body"].items():
            simplified_param_table[operation]["body"][str(body)] = body_values

    simplified_body_table = {}
    for operation, operation_values in body_obj_table.items():
        simplified_body_table[operation] = {}
        for mime_type, mime_values in operation_values.items():
            if mime_type not in simplified_body_table[operation]:
                simplified_body_table[operation][mime_type] = {}
            for body, body_values in mime_values.items():
                simplified_body_table[operation][mime_type][str(body)] = body_values

    compiled_q_table = {
        "OPERATION AGENT": operation_table,
        #"HEADER AGENT": header_table,
        "PARAMETER AGENT": simplified_param_table,
        "VALUE AGENT": value_table,
        "BODY OBJECT AGENT": simplified_body_table,
        "DATA SOURCE AGENT": data_source_table,
        "DEPENDENCY AGENT": dependency_table
    }
    output_dir = os.path.join(os.path.dirname(__file__), "src/data/completed_tables")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f"{output_dir}/{spec_name}.json", "w") as f:
        json.dump(compiled_q_table, f, indent=2)


class AutoRestTest:
    def __init__(self, spec_dir: str, local_test: bool, is_naive=False):
        self.spec_dir = spec_dir
        self.local_test = local_test
        self.is_naive = is_naive
        _construct_db_dir()
        self.db_q_table = os.path.join(os.path.dirname(__file__), "src/data/q_table")
        self.db_graph = os.path.join(os.path.dirname(__file__), "src/data/graph")
        self.use_cached_graph = False
        self.use_cached_table = False
        self.use_cached_values = True
        self.use_cached_headers = True

    def init_graph(self, spec_name: str, spec_path) -> OperationGraph:
        spec_parser = SpecificationParser(spec_path=spec_path, spec_name=spec_name)
        api_url = get_api_url(spec_parser, self.local_test)
        operation_graph = OperationGraph(spec_path=spec_path, spec_name=spec_name, spec_parser=spec_parser)
        request_generator = RequestGenerator(operation_graph=operation_graph, api_url=api_url, is_naive=self.is_naive)
        operation_graph.assign_request_generator(request_generator)
        return operation_graph

    def generate_graph(self, spec_name: str):
        spec_path = f"{self.spec_dir}/{spec_name}.yaml"
        print("CREATING GRAPH...")
        with shelve.open(self.db_graph) as db:
            if spec_name in db and self.use_cached_graph:
                operation_graph = self.init_graph(spec_name, spec_path)
                graph_properties = db[spec_name]
                operation_graph.operation_edges = graph_properties["edges"]
                operation_graph.operation_nodes = graph_properties["nodes"]
                print(f"Loaded graph for {spec_name} from shelve.")
            else:
                operation_graph = self.init_graph(spec_name, spec_path)
                operation_graph.create_graph()
                graph_properties = {
                    "edges": operation_graph.operation_edges,
                    "nodes": operation_graph.operation_nodes
                }
                try:
                    db[spec_name] = graph_properties
                except Exception as e:
                    print("Error saving graph to shelve.")
                print(f"Initialized new graph for {spec_name}.")
        print("GRAPH CREATED!!!")
        #operation_graph.print_graph()
        return operation_graph

    def perform_q_learning(self, operation_graph: OperationGraph, spec_name: str):
        print("BEGINNING Q-LEARNING...")
        q_learning = QLearning(operation_graph, alpha=0.1, gamma=0.9, epsilon=0.3, time_duration=1800, mutation_rate=0.25)
        with shelve.open(self.db_q_table) as db:
            if spec_name in db and self.use_cached_table:
                compiled_q_table = db[spec_name]
                #if self.use_cached_headers:
                #    q_learning.header_agent.q_table = compiled_q_table["header"]
                #else:
                #    q_learning.header_agent.initialize_q_table()
                #    db[spec_name]["header"] = q_learning.header_agent.q_table
                if self.use_cached_values:
                    q_learning.value_agent.q_table = compiled_q_table["value"]
                else:
                    q_learning.value_agent.initialize_q_table()
                    try:
                        db[spec_name]["value"] = q_learning.value_agent.q_table
                    except Exception as e:
                        print("Error saving value agent to shelve.")
                print(f"Loaded Q-table for {spec_name} from shelve.")
            else:
                q_learning.initialize_llm_agents()
                compiled_q_table = {
                #    "header": q_learning.header_agent.q_table,
                    "value": q_learning.value_agent.q_table
                }
                try:
                    db[spec_name] = compiled_q_table
                except:
                    print("Error saving Q-table to shelve.")
                print(f"Initialized new Q-tables for {spec_name}.")
        q_learning.parameter_agent.initialize_q_table()
        q_learning.operation_agent.initialize_q_table()
        q_learning.body_object_agent.initialize_q_table()
        q_learning.dependency_agent.initialize_q_table()
        q_learning.data_source_agent.initialize_q_table()
        q_learning.run()
        print("Q-LEARNING COMPLETED!!!")
        return q_learning

    def override_header_agent_q_table(self, operation_graph: OperationGraph, spec_name: str):
        q_learning = QLearning(operation_graph, alpha=0.1, gamma=0.9, epsilon=0.2, time_duration=1200)
        with shelve.open(self.db_q_table) as db:
            q_learning.header_agent.initialize_q_table()
            db[spec_name]["header"] = q_learning.header_agent.q_table

    def print_performance(self):
        print("Total cost of the tool: $", OpenAILanguageModel.get_cumulative_cost())

    def run_all(self):
        for spec in os.listdir(self.spec_dir):
            spec_name = spec.split(".")[0]
            print(f"Running tests for {spec_name}")
            self.run_single(spec_name)

    def run_single(self, spec_name: str):
        print("BEGINNING AUTO-REST-TEST...")
        operation_graph = self.generate_graph(spec_name)
        q_learning = self.perform_q_learning(operation_graph, spec_name)
        #self.override_header_agent_q_table(operation_graph, spec_name)
        self.print_performance()
        output_q_table(q_learning, spec_name)
        print("AUTO-REST-TEST COMPLETED!!!")

if __name__ == "__main__":
    # example of running: python3 AutoRestTest.py one true -s genome-nexus
    #spec_dir = "specs/original/oas"
    spec_dir = "aratrl-openapi/"
    args = parse_args()
    auto_rest_test = AutoRestTest(spec_dir=spec_dir, local_test=args.local_test, is_naive=False)
    if args.num_specs == "one":
        auto_rest_test.run_single(args.spec_name)
    else:
        auto_rest_test.run_all()