import argparse
import json
import os
import shelve

from src.generate_graph import OperationGraph
from src.graph.similarity_comparator import OperationDependencyComparator
from src.marl import QLearning
from src.request_generator import RequestGenerator

from dotenv import load_dotenv

from src.graph.specification_parser import SpecificationParser
from src.utils import OpenAILanguageModel, construct_db_dir

from configurations import USE_CACHED_GRAPH, USE_CACHED_TABLE, \
    LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_RATE, TIME_DURATION, MUTATION_RATE, SPECIFICATION_LOCATION, \
    OPENAI_LLM_ENGINE

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

def output_q_table(q_learning: QLearning, spec_name):
    parameter_table = q_learning.parameter_agent.q_table
    body_obj_table = q_learning.body_object_agent.q_table
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
    output_dir = os.path.join(os.path.dirname(__file__), f"data/{spec_name}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f"{output_dir}/q_tables.json", "w") as f:
        json.dump(compiled_q_table, f, indent=2)

def output_successes(q_learning: QLearning, spec_name: str):
    output_dir = os.path.join(os.path.dirname(__file__), f"data/{spec_name}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f"{output_dir}/successful_parameters.json", "w") as f:
        json.dump(q_learning.successful_parameters, f, indent=2)

    with open(f"{output_dir}/successful_bodies.json", "w") as f:
        json.dump(q_learning.successful_bodies, f, indent=2)

    with open(f"{output_dir}/successful_responses.json", "w") as f:
        json.dump(q_learning.successful_responses, f, indent=2)

    with open(f"{output_dir}/successful_primitives.json", "w") as f:
        json.dump(q_learning.successful_primitives, f, indent=2)

def output_errors(q_learning: QLearning, spec_name: str):
    output_dir = os.path.join(os.path.dirname(__file__), f"data/{spec_name}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f"{output_dir}/server_errors.json", "w") as f:
        json.dump(q_learning.errors, f, indent=2)

def output_operation_status_codes(q_learning: QLearning, spec_name: str):
    output_dir = os.path.join(os.path.dirname(__file__), f"data/{spec_name}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f"{output_dir}/operation_status_codes.json", "w") as f:
        json.dump(q_learning.operation_response_counter, f, indent=2)

def output_report(q_learning: QLearning, spec_name: str, spec_parser: SpecificationParser):
    output_dir = os.path.join(os.path.dirname(__file__), f"data/{spec_name}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    title = spec_parser.get_api_title() if spec_parser.get_api_title() else spec_name

    unique_processed_200s = set()
    for operation_idx, status_codes in q_learning.operation_response_counter.items():
        for status_code in status_codes:
            if status_code // 100 == 2:
                unique_processed_200s.add(operation_idx)

    report_content = {
        "Title": "AutoRestTest Report for " + title,
        "Duration": f"{q_learning.time_duration} seconds",
        "Status Code Distribution": dict(q_learning.responses),
        "Number of Total Operations": len(q_learning.operation_agent.q_table),
        "Number of Successfully Processed Operations": len(unique_processed_200s),
        "Percentage of Successfully Processed Operations": str(round(len(unique_processed_200s) / len(q_learning.operation_agent.q_table) * 100, 2)) + "%",
        "Number of Server Errors": len(q_learning.errors),
        "Number of Unique Server Errors": len(set(q_learning.errors)),
    }

    with open(f"{output_dir}/report.json", "w") as f:
        json.dump(report_content, f, indent=2)


def parse_specification_location(spec_loc: str):
    directory, file_name = os.path.split(spec_loc)
    file_name, ext = os.path.splitext(file_name)
    return directory, file_name, ext

class AutoRestTest:
    def __init__(self, spec_dir: str):
        self.spec_dir = spec_dir
        self.local_test = True
        self.is_naive = False
        construct_db_dir()
        self.use_cached_graph = USE_CACHED_GRAPH
        self.use_cached_table = USE_CACHED_TABLE

    def init_graph(self, spec_name: str, spec_path) -> OperationGraph:
        spec_parser = SpecificationParser(spec_path=spec_path, spec_name=spec_name)
        api_url = get_api_url(spec_parser, self.local_test)
        operation_graph = OperationGraph(spec_path=spec_path, spec_name=spec_name, spec_parser=spec_parser)
        request_generator = RequestGenerator(operation_graph=operation_graph, api_url=api_url, is_naive=self.is_naive)
        operation_graph.assign_request_generator(request_generator)
        return operation_graph

    def generate_graph(self, spec_name: str, ext: str):
        spec_path = f"{self.spec_dir}/{spec_name}{ext}"
        db_graph = os.path.join(os.path.dirname(__file__), f"src/data/graphs/{spec_name}_graph")
        print("CREATING GRAPH...")
        with shelve.open(db_graph) as db:

            loaded_from_shelf = False
            if spec_name in db and self.use_cached_graph:
                print(f"Loading graph for {spec_name} from shelve.")
                operation_graph = self.init_graph(spec_name, spec_path)

                try:
                    graph_properties = db[spec_name]
                    operation_graph.operation_edges = graph_properties["edges"]
                    operation_graph.operation_nodes = graph_properties["nodes"]
                    print(f"Loaded graph for {spec_name} from shelve.")
                    loaded_from_shelf = True
                except Exception as e:
                    print("Error loading graph from shelve.")
                    loaded_from_shelf = False

            if not loaded_from_shelf:
                print(f"Initializing new graph for {spec_name}.")
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
        return operation_graph

    def perform_q_learning(self, operation_graph: OperationGraph, spec_name: str):
        print("BEGINNING Q-LEARNING...")
        q_learning = QLearning(operation_graph, alpha=LEARNING_RATE, gamma=DISCOUNT_FACTOR, epsilon=EXPLORATION_RATE, time_duration=TIME_DURATION, mutation_rate=MUTATION_RATE)
        db_q_table = os.path.join(os.path.dirname(__file__), f"src/data/q_tables/{spec_name}_q_table")

        with shelve.open(db_q_table) as db:
            loaded_from_shelf = False

            if spec_name in db and self.use_cached_table:
                compiled_q_table = db[spec_name]
                q_learning.value_agent.q_table = compiled_q_table["value"]

                try:
                    db[spec_name]["value"] = q_learning.value_agent.q_table
                    print(f"Loaded Q-table for {spec_name} from shelve.")
                    loaded_from_shelf = True
                except Exception as e:
                    print("Error saving value agent to shelve.")
                    loaded_from_shelf = False

            if not loaded_from_shelf:
                q_learning.initialize_llm_agents()

                compiled_q_table = {
                    "value": q_learning.value_agent.q_table
                }

                try:
                    db[spec_name] = compiled_q_table
                except Exception as e:
                    print("Error saving Q-table to shelve.")

                print(f"Initialized new Q-tables for {spec_name}.")

        q_learning.parameter_agent.initialize_q_table()
        q_learning.operation_agent.initialize_q_table()
        q_learning.body_object_agent.initialize_q_table()
        q_learning.dependency_agent.initialize_q_table()
        q_learning.data_source_agent.initialize_q_table()
        output_q_table(q_learning, spec_name)
        q_learning.run()
        print("Q-LEARNING COMPLETED!!!")
        return q_learning

    def print_performance(self):
        print("Total cost of the tool: $", OpenAILanguageModel.get_cumulative_cost())

    def run_all(self):
        for spec in os.listdir(self.spec_dir):
            spec_name = spec.split(".")[0]
            print(f"Running tests for {spec_name}")
            self.run_single(spec_name)

    def run_single(self, spec_name: str, ext: str):
        print("BEGINNING AUTO-REST-TEST...")
        operation_graph = self.generate_graph(spec_name, ext)
        q_learning = self.perform_q_learning(operation_graph, spec_name)
        self.print_performance()
        output_q_table(q_learning, spec_name)
        output_successes(q_learning, spec_name)
        output_errors(q_learning, spec_name)
        output_operation_status_codes(q_learning, spec_name)
        output_report(q_learning, spec_name, operation_graph.spec_parser)
        print("AUTO-REST-TEST COMPLETED!!!")

if __name__ == "__main__":
    specification_directory, specification_name, ext = parse_specification_location(SPECIFICATION_LOCATION)
    auto_rest_test = AutoRestTest(spec_dir=specification_directory)
    auto_rest_test.run_single(specification_name, ext)