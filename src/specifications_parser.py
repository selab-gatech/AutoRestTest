from typing import List, Dict, Optional, Union
from prance import ResolvingParser, BaseParser
from dataclasses import dataclass, field

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
class BaseOperationProperties:
    operation_id: str = ''
    endpoint_path: str = ''
    http_method: str = ''

@dataclass
class OperationProperties(BaseOperationProperties):
    parameters: Dict[str, ParameterProperties] = field(default_factory=dict) # keep a mapping of name to properties

@dataclass
class RequestBodyOperationProperties(BaseOperationProperties):
    mime_type: str = ''
    properties: Dict[str, ValueProperties] = field(default_factory=dict)

class SpecificationParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.resolving_parser = ResolvingParser(file_path, strict=False)
        self.base_parser = BaseParser(file_path, strict=False)

    # will return operationID, and operation values
    def parse_specification(self) -> Dict[str, BaseOperationProperties]:
        spec_paths = self.resolving_parser.specification.get('paths', {})
