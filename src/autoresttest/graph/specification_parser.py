import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

from prance import ResolvingParser, BaseParser

from autoresttest.models import (
    OperationProperties,
    ParameterProperties,
    ResponseProperties,
    SchemaProperties,
    to_dict_helper,
)


def json_spec_output(output_directory: Path, file_name: str, spec: Dict):
    """
    Create a testing JSON file from the specification parsing output.
    """
    output_file = output_directory / file_name
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(spec, file, ensure_ascii=False, indent=4)


class SpecificationParser:
    """
    Class to parse a specification file and return a dictionary of all the operations and their properties.
    """

    def __init__(self, spec_path=None, spec_name=None):
        self.spec_path = spec_path
        self.spec_name = spec_name
        if spec_path is not None:
            self.resolving_parser = ResolvingParser(spec_path, strict=False)
            self.base_parser = BaseParser(spec_path, strict=False)
        else:
            raise ValueError("No specification path provided.")
        # self.directory_path = 'specs/original/oas/'
        self.directory_path = "specs/aratrl-openapi/"
        self.all_specs = {}

    def get_api_url(self) -> str:
        """
        Extract the server URL from the specification file.
        """
        return self.resolving_parser.specification.get("servers")[0].get("url")

    def get_api_title(self) -> str:
        """
        Extract the title of the API from the specification file.
        """
        return self.resolving_parser.specification.get("info", {}).get("title")

    def process_parameter_object_properties(
        self, properties: Dict
    ) -> Dict[str, SchemaProperties]:
        """
        Process the properties of a parameter of type object to return a dictionary of all the properties and their
        corresponding parameter values.
        """
        if properties is None:
            return None
        object_properties = {}
        for name, values in properties.items():
            object_properties.setdefault(
                name, self.process_parameter_schema(values)
            )  # check if this is correct, or if it should be process_parameter
        return object_properties

    def process_parameter_schema(
        self, schema: Dict, description: str = None
    ) -> SchemaProperties:
        """
        Process the schema of a parameter to return a ValueProperties object
        """
        if not schema:
            return None

        value_properties = SchemaProperties(
            type=schema.get("type"),
            format=schema.get("format"),
            description=schema.get("description") if not description else description,
            items=self.process_parameter_schema(
                schema.get("items")
            ),  # recursively process items
            properties=self.process_parameter_object_properties(
                schema.get("properties")
            ),
            required=schema.get("required"),
            default=schema.get("default"),
            enum=schema.get("enum"),
            minimum=schema.get("minimum"),
            maximum=schema.get("maximum"),
            min_length=schema.get("minLength"),
            max_length=schema.get("maxLength"),
            pattern=schema.get("pattern"),
            max_items=schema.get("maxItems"),
            min_items=schema.get("minItems"),
            unique_items=schema.get("uniqueItems"),
            additional_properties=schema.get("additionalProperties"),
            nullable=schema.get("nullable"),
            read_only=schema.get("readOnly"),
            write_only=schema.get("writeOnly"),
            example=schema.get("example"),
            examples=schema.get("examples"),
        )
        return value_properties

    def process_parameter(self, parameter) -> ParameterProperties:
        """
        Process an individual parameter to return a ParameterProperties object.
        """
        parameter_properties = ParameterProperties(
            name=parameter.get("name"),
            in_value=parameter.get("in"),
            description=parameter.get("description"),
            required=parameter.get("required"),
            deprecated=parameter.get("deprecated"),
            allow_empty_value=parameter.get("allowEmptyValue"),
            style=parameter.get("style"),
            explode=parameter.get("explode"),
            allow_reserved=parameter.get("allowReserved"),
        )
        if parameter.get("schema"):
            parameter_properties.schema = self.process_parameter_schema(
                parameter.get("schema")
            )
        return parameter_properties

    def process_parameters(self, parameter_list) -> Dict[str, ParameterProperties]:
        """
        Process the parameters list to return a Dictionary with all its properties and values.
        """
        parameters = {}
        if parameter_list:
            for parameter in parameter_list:
                parameter_properties = self.process_parameter(parameter)
                parameters.setdefault(parameter_properties.name, parameter_properties)
        return parameters

    def process_request_body(self, request_body) -> Dict[str, SchemaProperties]:
        """
        Process the request body to return a Dictionary with mime type and its properties and values.
        """

        request_body_properties = {}
        content = request_body.get("content")
        description = request_body.get("description")
        if content:
            for mime_type, mime_details in content.items():
                # if we need to check required list, do it here
                schema = mime_details.get("schema")
                if schema:
                    request_body_properties[mime_type] = self.process_parameter_schema(
                        schema, description
                    )

        return request_body_properties

    def process_responses(self, responses) -> Dict[str, ResponseProperties]:
        """
        Process the responses to return a Dictionary with status code and its properties and values.
        """
        response_properties = {}
        for status_code, response_details in responses.items():
            response_properties.setdefault(
                status_code,
                ResponseProperties(
                    status_code=status_code,
                    description=response_details.get("description"),
                ),
            )
            content = response_details.get("content")
            if content:
                for mime_type, mime_details in content.items():
                    # if we need to check required list, do it here
                    schema = mime_details.get("schema")
                    if schema:
                        response_properties[status_code].content[mime_type] = (
                            self.process_parameter_schema(schema)
                        )
        return response_properties

    def process_operation_details(
        self, http_method: str, endpoint_path: str, operation_details: Dict
    ) -> OperationProperties:
        """
        Process the parameters and request body details within a given operation to return as OperationProperties object.
        """
        operation_properties = OperationProperties(
            operation_id=operation_details.get("operationId"),
            endpoint_path=endpoint_path,
            http_method=http_method,
            summary=operation_details.get("summary"),
        )
        if operation_details.get("parameters"):
            operation_properties.parameters = self.process_parameters(
                parameter_list=operation_details.get("parameters")
            )

        if operation_details.get("requestBody"):
            operation_properties.request_body = self.process_request_body(
                request_body=operation_details.get("requestBody")
            )

        if operation_details.get("responses"):
            operation_properties.responses = self.process_responses(
                responses=operation_details.get("responses")
            )

        # maybe add security details?

        return operation_properties

    def parse_specification(self) -> Dict[str, OperationProperties]:
        """
        Parse the specification file to return a dictionary of all the operations and their properties.

        The key of the dictionary is the operationId and the value is an OperationProperties object.
        """
        supported_methods = {"get", "post", "put", "delete", "head", "options", "patch"}
        operation_collection = {}
        spec_paths = self.resolving_parser.specification.get("paths", {})
        for endpoint_path, endpoint_details in spec_paths.items():
            for http_method, operation_details in endpoint_details.items():
                if http_method in supported_methods:
                    operation_properties = self.process_operation_details(
                        http_method, endpoint_path, operation_details
                    )
                    operation_collection.setdefault(
                        operation_properties.operation_id, operation_properties
                    )
        return operation_collection

    def parse_all_specifications(self) -> dict:
        """
        Parse all the specification files in the directory to return a dictionary of all the operations and their properties.
        """
        for file_name in os.listdir(self.directory_path):
            print("Specification parsing for file: ", file_name)
            self.spec_path = os.path.join(self.directory_path, file_name)
            self.resolving_parser = ResolvingParser(self.spec_path, strict=False)
            self.all_specs[file_name] = self.parse_specification()
            # print("Output: " + str(output))

        return self.all_specs

    def all_json_spec_output(self):
        """
        Create a testing JSON file from the specification parsing output.
        """
        all_specs = {
            file_name: to_dict_helper(spec)
            for file_name, spec in self.all_specs.items()
        }

        output_directory = Path("./testing_output")
        output_directory.mkdir(parents=True, exist_ok=True)

        for file_name, spec in all_specs.items():
            json_spec_output(output_directory, file_name, spec)

    def single_json_spec_output(self):
        output_directory = Path("./testing_output")
        output_directory.mkdir(parents=True, exist_ok=True)
        output = self.parse_specification()
        json_spec_output(output_directory, self.spec_name, to_dict_helper(output))


if __name__ == "__main__":
    # testing
    spec_path = os.path.normpath("../../aratrl-openapi/project.yaml")
    spec_parser = SpecificationParser(spec_name="project", spec_path=spec_path)
    spec_parser.single_json_spec_output()
    # spec_parser.parse_all_specifications()
    # spec_parser.all_json_spec_output()

    # spec_parser = SpecificationParser(spec_path="../specs/original/oas/genome-nexus.yaml", spec_name="genome-nexus")
    # spec_parser.single_json_spec_output()

    # output = spec_parser.parse_specification()
    # print(output["fetchPostTranslationalModificationsByPtmFilterPOST"])
    # print(output["endpoint-add-tracks-to-playlist"])
    # print(output["endpoint-get-playlist"])
    # print(output["endpoint-remove-tracks-playlist"])
