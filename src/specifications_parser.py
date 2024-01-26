from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field

from prance import ResolvingParser, BaseParser

@dataclass
class ValueProperties:
    type: Optional[str] = None
    format: Optional[str] = None
    description: Optional[str] = None
    items: 'ValueProperties' = field(default_factory=lambda: ValueProperties())
    properties: Dict[str, 'ValueProperties'] = field(default_factory=dict)
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
    additional_properties: Union[bool, 'ValueProperties', None] = True
    nullable: Optional[bool] = None
    read_only: Optional[bool] = None
    write_only: Optional[bool] = None
    example: Optional[Union[str, int, float, bool, List, Dict]] = None
    examples: List[Optional[Union[str, int, float, bool, List, Dict]]] = field(default_factory=list)

@dataclass
class ParameterProperties:
    name: str = ''
    in_value: Optional[str] = None
    description: Optional[str] = None
    required: Optional[bool] = None
    deprecated: Optional[bool] = None
    allow_empty_value: Optional[bool] = None
    style: Optional[str] = None
    explode: Optional[bool] = None
    allow_reserved: Optional[bool] = None
    schema: ValueProperties = field(default_factory=lambda: ValueProperties())
    example: Optional[Union[str, int, float, bool, List, Dict]] = None
    examples: List[Optional[Union[str, int, float, bool, List, Dict]]] = field(default_factory=list)
    content: Dict[str, ValueProperties] = field(default_factory=dict)

@dataclass
class OperationProperties:
    operation_id: str = ''
    endpoint_path: str = ''
    http_method: str = ''
    parameters: Dict[str, ParameterProperties] = field(default_factory=dict)
    request_body: bool = False
    request_body_properties: Dict[str, ValueProperties] = field(default_factory=dict)

class SpecificationParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.resolving_parser = ResolvingParser(file_path, strict=False)
        self.base_parser = BaseParser(file_path, strict=False)

    def process_request_body(self, request_body) -> Dict[str, ValueProperties]:
        """
        Process the request body to return a Dictionary with all its properties and values
        """
        return None

    def process_operation_details(self, http_method: str, endpoint_path: str, operation_details: Dict) -> OperationProperties:
        """
        Process the parameters and request body details within a given operation to return as OperationProperties object
        """
        operation_properties = OperationProperties(
            operation_id=operation_details.get('operationId'),
            endpoint_path=endpoint_path,
            http_method=http_method
        )

        parameter = operation_details.get('parameters')

        if operation_details.get('requestBody'):
            operation_properties.request_body = True
            operation_properties.request_body_properties = self.process_request_body(operation_details.get('requestBody'))

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
    spec_parser = SpecificationParser("../specs/original/oas/youtube.yaml")
    print(spec_parser.parse_specification())
