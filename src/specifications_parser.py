from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field
import json

from prance import ResolvingParser, BaseParser

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
    request_body_properties: Dict[str, Dict[str, ItemProperties]] = None # MIME type as first key, then each parameter with its properties as second dict

class SpecificationParser:
    """
    Class to parse a specification file and return a dictionary of all the operations and their properties.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.resolving_parser = ResolvingParser(file_path, strict=False)
        self.base_parser = BaseParser(file_path, strict=False)

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
        for parameter in parameter_list:
            parameter_properties = self.process_parameter(parameter)
            parameters.setdefault(parameter_properties.name, parameter_properties)
        return parameters

    def process_request_body(self, request_body) -> Dict[str, Dict[str, ItemProperties]]:
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
                    properties = schema.get('properties')
                    if properties:
                        request_body_properties[mime_type] = self.process_parameter_object_properties(properties)
                    else:
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

        operation_properties.parameters = self.process_parameters(operation_details.get('parameters'))

        if operation_details.get('requestBody'):
            operation_properties.request_body = True
            operation_properties.request_body_properties = self.process_request_body(operation_details.get('requestBody'))
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

if __name__ == "__main__":
    # testing
    spec_parser = SpecificationParser("../specs/original/oas/spotify.yaml")
    output = spec_parser.parse_specification()
    print(output["endpoint-add-tracks-to-playlist"])
    print(output["endpoint-get-playlist"])
    print(output["endpoint-remove-tracks-playlist"])
    #spec_parser.parse_specification()
