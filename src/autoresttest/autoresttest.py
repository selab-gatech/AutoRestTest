import argparse
import json
import shelve
from pathlib import Path
from typing import Union

from autoresttest.graph.generate_graph import OperationGraph
from autoresttest.marl import QLearning
from autoresttest.graph import RequestGenerator

from dotenv import load_dotenv

from autoresttest.llm import OpenAILanguageModel
from autoresttest.specification import SpecificationParser
from autoresttest.utils import (
    construct_db_dir,
    is_json_seriable,
    EmbeddingModel,
    get_api_url,
    get_graph_cache_path,
    get_q_table_cache_path,
)
from autoresttest.models import to_dict_helper

from autoresttest.config import get_config

load_dotenv()

AUTORESTTEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AUTORESTTEST_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

CONFIG = get_config()


def ensure_output_dir(spec_name: str) -> Path:
    output_dir = DATA_ROOT / spec_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate requests based on API specification."
    )
    parser.add_argument(
        "num_specs",
        choices=["one", "many"],
        help="Specifies the number of specifications: 'one' or 'many'",
    )
    parser.add_argument(
        "local_test",
        type=lambda x: (str(x).lower() == "true"),
        help="Specifies whether the test is local (true/false)",
    )
    parser.add_argument(
        "-s",
        "--spec_name",
        type=str,
        default=None,
        help="Optional name of the specification",
    )
    return parser.parse_args()


def output_q_table(q_learning: QLearning, spec_name):
    parameter_table = q_learning.parameter_agent.q_table
    body_obj_table = q_learning.body_object_agent.q_table
    value_table = q_learning.value_agent.q_table
    operation_table = q_learning.operation_agent.q_table
    data_source_table = q_learning.data_source_agent.q_table
    dependency_table = q_learning.dependency_agent.q_table
    header_table = (
        q_learning.header_agent.q_table
        if q_learning.header_agent.q_table
        else "Disabled"
    )

    simplified_param_table = {}
    for operation, operation_values in parameter_table.items():
        simplified_param_table[operation] = {"params": {}, "body": {}}
        for parameter, parameter_values in operation_values["params"].items():
            simplified_param_table[operation]["params"][
                str(parameter)
            ] = parameter_values
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
        "HEADER AGENT": header_table,
        "PARAMETER AGENT": simplified_param_table,
        "VALUE AGENT": value_table,
        "BODY OBJECT AGENT": simplified_body_table,
        "DATA SOURCE AGENT": data_source_table,
        "DEPENDENCY AGENT": dependency_table,
    }
    compiled_q_table = to_dict_helper(compiled_q_table)
    output_dir = ensure_output_dir(spec_name)

    q_tables_path = output_dir / "q_tables.json"
    with q_tables_path.open("w") as f:
        json.dump(compiled_q_table, f, indent=2)


def output_successes(q_learning: QLearning, spec_name: str):
    output_dir = ensure_output_dir(spec_name)

    with (output_dir / "successful_parameters.json").open("w") as f:
        json.dump(to_dict_helper(q_learning.successful_parameters), f, indent=2)

    with (output_dir / "successful_bodies.json").open("w") as f:
        json.dump(q_learning.successful_bodies, f, indent=2)

    with (output_dir / "successful_responses.json").open("w") as f:
        json.dump(q_learning.successful_responses, f, indent=2)

    with (output_dir / "successful_primitives.json").open("w") as f:
        json.dump(q_learning.successful_primitives, f, indent=2)


def output_errors(q_learning: QLearning, spec_name: str):
    output_dir = ensure_output_dir(spec_name)

    seriable_errors = {}
    for operation_idx, unique_errors in q_learning.unique_errors.items():
        seriable_errors[operation_idx] = [
            error for error in unique_errors if is_json_seriable(error)
        ]

    with (output_dir / "server_errors.json").open("w") as f:
        json.dump(seriable_errors, f, indent=2)


def output_operation_status_codes(q_learning: QLearning, spec_name: str):
    output_dir = ensure_output_dir(spec_name)

    with (output_dir / "operation_status_codes.json").open("w") as f:
        json.dump(q_learning.operation_response_counter, f, indent=2)


def output_report(
    q_learning: QLearning, spec_name: str, spec_parser: SpecificationParser
):
    output_dir = ensure_output_dir(spec_name)

    title = spec_parser.get_api_title() if spec_parser.get_api_title() else spec_name
    title = f"'{title}' ({spec_name})"

    unique_processed_200s = set()
    for operation_idx, status_codes in q_learning.operation_response_counter.items():
        for status_code in status_codes:
            if status_code // 100 == 2:
                unique_processed_200s.add(operation_idx)

    unique_errors = 0
    for operation_idx in q_learning.unique_errors:
        unique_errors += len(q_learning.unique_errors[operation_idx])

    total_requests = sum(q_learning.responses.values())

    report_content = {
        "Title": "AutoRestTest Report for " + title,
        "Duration": f"{q_learning.time_duration} seconds",
        "Total Requests Sent": total_requests,
        "Status Code Distribution": dict(q_learning.responses),
        "Number of Total Operations": len(q_learning.operation_agent.q_table),
        "Number of Successfully Processed Operations": len(unique_processed_200s),
        "Percentage of Successfully Processed Operations": str(
            round(
                len(unique_processed_200s)
                / len(q_learning.operation_agent.q_table)
                * 100,
                2,
            )
        )
        + "%",
        "Number of Unique Server Errors": unique_errors,
        "Operations with Server Errors": q_learning.errors,
    }

    with (output_dir / "report.json").open("w") as f:
        json.dump(report_content, f, indent=2)


def parse_specification_location(spec_loc: str):
    spec_path = Path(spec_loc).expanduser()
    return spec_path.parent, spec_path.stem, spec_path.suffix


class AutoRestTest:
    def __init__(self, spec_dir: Union[Path, str]):
        self.spec_dir = Path(spec_dir).expanduser()
        self.local_test = True
        self.is_naive = False
        construct_db_dir()
        self.use_cached_graph = CONFIG.cache.use_cached_graph
        self.use_cached_table = CONFIG.cache.use_cached_table

    def init_graph(
        self,
        spec_name: str,
        spec_path: Union[Path, str],
        embedding_model: EmbeddingModel,
    ) -> OperationGraph:
        print(f"Parsing OpenAPI specification: {spec_path}...")
        spec_parser = SpecificationParser(spec_path=str(spec_path), spec_name=spec_name)
        print("Specification parsed successfully!")
        api_url = get_api_url(spec_parser, self.local_test)
        operation_graph = OperationGraph(
            spec_path=str(spec_path),
            spec_name=spec_name,
            spec_parser=spec_parser,
            embedding_model=embedding_model,
        )
        request_generator = RequestGenerator(
            operation_graph=operation_graph, api_url=api_url, is_naive=self.is_naive
        )
        operation_graph.assign_request_generator(request_generator)
        return operation_graph

    def generate_graph(self, spec_name: str, ext: str, embedding_model: EmbeddingModel):
        spec_path = self.spec_dir / f"{spec_name}{ext}"
        db_graph = get_graph_cache_path(spec_name)
        print("CREATING SEMANTIC OPERATION DEPENDECY GRAPH...")
        with shelve.open(str(db_graph)) as db:

            loaded_from_shelf = False
            if spec_name in db and self.use_cached_graph:
                print(f"Loading graph for {spec_name} from shelve.")
                operation_graph = self.init_graph(spec_name, spec_path, embedding_model)

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
                operation_graph = self.init_graph(spec_name, spec_path, embedding_model)
                operation_graph.create_graph()

                graph_properties = {
                    "edges": operation_graph.operation_edges,
                    "nodes": operation_graph.operation_nodes,
                }

                try:
                    db[spec_name] = graph_properties
                except Exception as e:
                    print("Error saving graph to shelve.")

                print(f"Initialized new graph for {spec_name}.")
        print("GRAPH CREATED!!!")
        return operation_graph

    def perform_q_learning(self, operation_graph: OperationGraph, spec_name: str):
        print("INITIATING Q-TABLES...")
        q_learning = QLearning(
            operation_graph,
            alpha=CONFIG.q_learning.learning_rate,
            gamma=CONFIG.q_learning.discount_factor,
            epsilon=CONFIG.q_learning.max_exploration,
            time_duration=CONFIG.request_generation.time_duration,
            mutation_rate=CONFIG.request_generation.mutation_rate,
        )
        db_q_table = get_q_table_cache_path(spec_name)

        q_learning.operation_agent.initialize_q_table()
        print("Initialized operation agent Q-table.")
        q_learning.parameter_agent.initialize_q_table()
        print("Initialized parameter agent Q-table.")
        q_learning.body_object_agent.initialize_q_table()
        print("Initialized body object agent Q-table.")
        q_learning.dependency_agent.initialize_q_table()
        print("Initialized dependency agent Q-table.")
        q_learning.data_source_agent.initialize_q_table()
        print("Initialized data source agent Q-table.")

        output_q_table(q_learning, spec_name)

        with shelve.open(str(db_q_table)) as db:
            loaded_value_from_shelf = False
            loaded_header_from_shelf = False

            if spec_name in db and self.use_cached_table:
                print(f"Loading Q-tables for {spec_name} from shelve.")

                compiled_q_table = db[spec_name]

                try:
                    q_learning.value_agent.q_table = compiled_q_table["value"]
                    print(
                        f"Initialized value agent's Q-table for {spec_name} from shelve."
                    )
                    loaded_value_from_shelf = True
                except Exception as e:
                    print("Error loading value agent from shelve.")
                    loaded_value_from_shelf = False

                if CONFIG.enable_header_agent:
                    try:
                        q_learning.header_agent.q_table = compiled_q_table["header"]
                        print(
                            f"Initialized header agent's Q-table for {spec_name} from shelve."
                        )
                        loaded_header_from_shelf = (
                            True if q_learning.header_agent.q_table else False
                        )
                        # If the header agent is disabled, the Q-table will be None.
                    except Exception as e:
                        print("Error loading header agent from shelve.")
                        loaded_header_from_shelf = False

            if not loaded_value_from_shelf:
                q_learning.value_agent.initialize_q_table()
                print(f"Initialized new value agent Q-table for {spec_name}.")
                token_counter = OpenAILanguageModel.get_tokens()
                print(f"Value table generation tokens - Input: {token_counter.input_tokens}, Output: {token_counter.output_tokens}")

            if CONFIG.enable_header_agent and not loaded_header_from_shelf:
                q_learning.header_agent.initialize_q_table()
                print(f"Initialized new header agent Q-table for {spec_name}.")
                token_counter = OpenAILanguageModel.get_tokens()
                print(f"Header table generation tokens - Input: {token_counter.input_tokens}, Output: {token_counter.output_tokens}")
            elif not CONFIG.enable_header_agent:
                q_learning.header_agent.q_table = None

            try:
                db[spec_name] = {
                    "value": q_learning.value_agent.q_table,
                    "header": q_learning.header_agent.q_table,
                }
            except Exception as e:
                print("Error saving Q-tables to shelve.")

        output_q_table(q_learning, spec_name)
        print("Q-TABLES INITIALIZED...")

        print("BEGINNING Q-LEARNING...")
        q_learning.run()
        print("Q-LEARNING COMPLETED!!!")
        return q_learning

    def print_performance(self):
        token_counter = OpenAILanguageModel.get_tokens()
        print(f"Total input tokens used: {token_counter.input_tokens}")
        print(f"Total output tokens used: {token_counter.output_tokens}")

    def run_all(self):
        for spec_file in self.spec_dir.iterdir():
            if not spec_file.is_file():
                continue
            spec_name = spec_file.stem
            print(f"Running tests for {spec_name}")
            self.run_single(spec_name, spec_file.suffix)

    def run_single(self, spec_name: str, ext: str):
        print("BEGINNING AUTO-REST-TEST...")
        embedding_model = EmbeddingModel()
        operation_graph = self.generate_graph(spec_name, ext, embedding_model)
        q_learning = self.perform_q_learning(operation_graph, spec_name)
        self.print_performance()
        output_q_table(q_learning, spec_name)
        output_successes(q_learning, spec_name)
        output_errors(q_learning, spec_name)
        output_operation_status_codes(q_learning, spec_name)
        output_report(q_learning, spec_name, operation_graph.spec_parser)
        print("AUTO-REST-TEST COMPLETED!!!")


def main():
    specification_directory, specification_name, ext = parse_specification_location(
        PROJECT_ROOT / CONFIG.specification_location
    )
    auto_rest_test = AutoRestTest(spec_dir=specification_directory)
    auto_rest_test.run_single(specification_name, ext)


if __name__ == "__main__":
    main()
