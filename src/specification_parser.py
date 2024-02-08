import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Iterable
from dataclasses import dataclass, field
import json

from prance import ResolvingParser, BaseParser

def to_dict_helper(item):
    """
    Helper method for parsing in to a dictionary. Handles the case where the item is a dictionary, list, or object with
    a to_dict method.
    """
    if hasattr(item, 'to_dict'):
        return item.to_dict()
    elif isinstance(item, dict):
        return {k: to_dict_helper(v) for k, v in item.items()}
    elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
        return [to_dict_helper(i) for i in item]
    else:
        return item

@dataclass
class ItemProperties:
    """
    Class to store the properties of either the schema values, in the case of parameters, or the request body object values
    """
    type: Optional[str] = None
    format: Optional[str] = None
    description: Optional[str] = None
    items: 'ItemProperties' = None
    properties: Dict[str, 'ItemProperties'] = None
    required: List[str] = field(default_factory=list)
    default: Optional[Union[str, int, float, bool, List, Dict]] = None
    enum: Optional[List[str]] = field(default_factory=list)
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    max_items: Optional[int] = None
    min_items: Optional[int] = None
    unique_items: Optional[bool] = None
    additional_properties: Union[bool, 'ItemProperties', None] = True
    nullable: Optional[bool] = None
    read_only: Optional[bool] = None
    write_only: Optional[bool] = None
    example: Optional[Union[str, int, float, bool, List, Dict]] = None
    examples: List[Optional[Union[str, int, float, bool, List, Dict]]] = field(default_factory=list)

    def to_dict(self):
        result = {k: to_dict_helper(v) for k, v in self.__dict__.items() if v is not None}
        if 'items' in result and self.items is not None:
            result['items'] = self.items.to_dict()
        if 'properties' in result and self.properties:
            result['properties'] = {k: v.to_dict() for k, v in self.properties.items()}
        if 'additional_properties' in result and isinstance(self.additional_properties, ItemProperties):
            result['additional_properties'] = self.additional_properties.to_dict()
        return result

@dataclass
class ParameterProperties:
    """
    Class to store the properties of a parameter. Parameters have nested schemas, whereas request bodies do not.
    """
    name: str = ''
    in_value: Optional[str] = None
    description: Optional[str] = None
    required: Optional[bool] = None
    deprecated: Optional[bool] = None
    allow_empty_value: Optional[bool] = None
    style: Optional[str] = None
    explode: Optional[bool] = None
    allow_reserved: Optional[bool] = None
    schema: ItemProperties = None
    example: Optional[Union[str, int, float, bool, List, Dict]] = None
    examples: List[Optional[Union[str, int, float, bool, List, Dict]]] = field(default_factory=list)
    content: Dict[str, ItemProperties] = field(default_factory=dict)

    def to_dict(self):
        result = {k: to_dict_helper(v) for k, v in self.__dict__.items() if v is not None}
        if 'schema' in result and self.schema is not None:
            result['schema'] = self.schema.to_dict()
        if 'content' in result and self.content:
            result['content'] = {k: v.to_dict() for k, v in self.content.items()}
        return result

@dataclass
class OperationProperties:
    """
    Class to store the properties of an operation, considering both its parameters and potential request body.
    """
    operation_id: str = ''
    endpoint_path: str = ''
    http_method: str = ''
    parameters: Dict[str, ParameterProperties] = field(default_factory=dict)
    request_body: bool = False
    request_body_properties: Dict[str, ItemProperties] = None # MIME type as first key, then each parameter with its properties as second dict

    def to_dict(self):
        result = {k: to_dict_helper(v) for k, v in self.__dict__.items() if v is not None}
        if 'parameters' in result and self.parameters:
            result['parameters'] = {k: v.to_dict() for k, v in self.parameters.items()}
        if 'request_body_properties' in result and self.request_body_properties:
            result['request_body_properties'] = {k: v.to_dict() for k, v in self.request_body_properties.items()}
        return result

class SpecificationParser:
    """
    Class to parse a specification file and return a dictionary of all the operations and their properties.
    """
    def __init__(self, file_path=None):
        self.file_path = file_path
        if file_path is not None:
            self.resolving_parser = ResolvingParser(file_path, strict=False)
            self.base_parser = BaseParser(file_path, strict=False)
        self.directory_path = '../specs/original/oas/'
        self.all_specs = {}

    def process_parameter_object_properties(self, properties: Dict) -> Dict[str, ItemProperties]:
        """
        Process the properties of a parameter of type object to return a dictionary of all the properties and their
        corresponding parameter values.
        """
        if properties is None:
            return None

        object_properties = {}
        for name, values in properties.items():
            object_properties.setdefault(name, self.process_parameter_schema(values)) # check if this is correct, or if it should be process_parameter
        return object_properties

    def process_parameter_schema(self, schema: Dict) -> ItemProperties:
        """
        Process the schema of a parameter to return a ValueProperties object
        """

        if not schema:
            return None

        value_properties = ItemProperties(
            type=schema.get('type'),
            format=schema.get('format'),
            description=schema.get('description'),
            items=self.process_parameter_schema(schema.get('items')), # recursively process items
            properties=self.process_parameter_object_properties(schema.get('properties')),
            required=schema.get('required'),
            default=schema.get('default'),
            enum=schema.get('enum'),
            minimum=schema.get('minimum'),
            maximum=schema.get('maximum'),
            min_length=schema.get('minLength'),
            max_length=schema.get('maxLength'),
            pattern=schema.get('pattern'),
            max_items=schema.get('maxItems'),
            min_items=schema.get('minItems'),
            unique_items=schema.get('uniqueItems'),
            additional_properties=schema.get('additionalProperties'),
            nullable=schema.get('nullable'),
            read_only=schema.get('readOnly'),
            write_only=schema.get('writeOnly'),
            example=schema.get('example'),
            examples=schema.get('examples')
        )
        return value_properties

    def process_parameter(self, parameter) -> ParameterProperties:
        """
        Process an individual parameter to return a ParameterProperties object.
        """
        parameter_properties = ParameterProperties(
            name=parameter.get('name'),
            in_value=parameter.get('in'),
            description=parameter.get('description'),
            required=parameter.get('required'),
            deprecated=parameter.get('deprecated'),
            allow_empty_value=parameter.get('allowEmptyValue'),
            style=parameter.get('style'),
            explode=parameter.get('explode'),
            allow_reserved=parameter.get('allowReserved')
        )
        if parameter.get('schema'):
            parameter_properties.schema = self.process_parameter_schema(parameter.get('schema'))
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

    def process_request_body(self, request_body) -> Dict[str, ItemProperties]:
        """
        Process the request body to return a Dictionary with mime type and its properties and values.
        """

        request_body_properties = {}
        content = request_body.get('content')
        if content:
            for mime_type, mime_details in content.items():
                # if we need to check required list, do it here
                schema = mime_details.get('schema')
                if schema:
                    request_body_properties[mime_type] = self.process_parameter_schema(schema)

        return request_body_properties

    def process_operation_details(self, http_method: str, endpoint_path: str, operation_details: Dict) -> OperationProperties:
        """
        Process the parameters and request body details within a given operation to return as OperationProperties object.
        """
        operation_properties = OperationProperties(
            operation_id=operation_details.get('operationId'),
            endpoint_path=endpoint_path,
            http_method=http_method
        )
        if operation_details.get("parameters"):
            operation_properties.parameters = self.process_parameters(parameter_list=operation_details.get('parameters'))

        if operation_details.get('requestBody'):
            operation_properties.request_body = True
            operation_properties.request_body_properties = self.process_request_body(request_body=operation_details.get('requestBody'))
            # add response details if need-be here

        # maybe add security details?

        return operation_properties

    def parse_specification(self) -> Dict[str, OperationProperties]:
        """
        Parse the specification file to return a dictionary of all the operations and their properties.

        The key of the dictionary is the operationId and the value is an OperationProperties object.
        """
        operation_collection = {}
        spec_paths = self.resolving_parser.specification.get('paths', {})
        for endpoint_path, endpoint_details in spec_paths.items():
            for http_method, operation_details in endpoint_details.items():
                operation_properties = self.process_operation_details(http_method, endpoint_path, operation_details)
                operation_collection.setdefault(operation_properties.operation_id, operation_properties)
        return operation_collection

    def parse_all_specifications(self) -> dict:
        """
        Parse all the specification files in the directory to return a dictionary of all the operations and their properties.
        """
        for file_name in os.listdir(self.directory_path):
            print("Specification parsing for file: ", file_name)
            self.file_path = os.path.join(self.directory_path, file_name)
            self.resolving_parser = ResolvingParser(self.file_path, strict=False)
            self.all_specs[file_name]  = self.parse_specification()
            #print("Output: " + str(output))

        return self.all_specs

    def json_spec_output(self, output_directory: Path, file_name: str, spec: Dict):
        """
        Create a testing JSON file from the specification parsing output.
        """
        output_file = output_directory / file_name
        with output_file.open('w', encoding='utf-8') as file:
            json.dump(spec, file, ensure_ascii=False, indent=4)

    def all_json_spec_output(self):
        """
        Create a testing JSON file from the specification parsing output.
        """
        all_specs = {file_name: to_dict_helper(spec) for file_name, spec in self.all_specs.items()}

        output_directory = Path("./testing_output")
        output_directory.mkdir(parents=True, exist_ok=True)

        for file_name, spec in all_specs.items():
            self.json_spec_output(output_directory, file_name, spec)

if __name__ == "__main__":
    # testing
    spec_parser = SpecificationParser()
    spec_parser.parse_all_specifications()
    spec_parser.all_json_spec_output()
    #spec_parser = SpecificationParser("../specs/original/oas/genome-nexus.yaml")
    # output = spec_parser.parse_specification()
    #print(output["fetchPostTranslationalModificationsByPtmFilterPOST"])
    #print(output["endpoint-add-tracks-to-playlist"])
    #print(output["endpoint-get-playlist"])
    #print(output["endpoint-remove-tracks-playlist"])
    #spec_parser.parse_specification()
