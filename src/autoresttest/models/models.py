from __future__ import annotations

import itertools
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Union, TYPE_CHECKING

import requests

if TYPE_CHECKING:
    from autoresttest.graph.generate_graph import OperationEdge


def to_dict_helper(item: Any) -> Any:
    """Convert nested dataclasses, dictionaries, and iterables into serialisable structures."""
    if hasattr(item, "to_dict"):
        return item.to_dict()
    if isinstance(item, dict):
        converted = {
            key: to_dict_helper(value)
            for key, value in item.items()
            if value is not None
        }
        return {key: value for key, value in converted.items() if value is not None}
    if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
        converted = [to_dict_helper(element) for element in item]
        return [value for value in converted if value is not None]
    return item


@dataclass
class ResponseProperties:
    """Stores the properties of an HTTP response."""

    status_code: int = -1
    description: Optional[str] = None
    content: Dict[str, "SchemaProperties"] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_dict_helper(asdict(self))


@dataclass
class SchemaProperties:
    """Represents schema properties for parameters or request bodies."""

    type: Optional[str] = None
    format: Optional[str] = None
    description: Optional[str] = None
    items: Optional["SchemaProperties"] = None
    properties: Optional[Dict[str, "SchemaProperties"]] = None
    required: List[str] = field(default_factory=list)
    default: Optional[Union[str, int, float, bool, List, Dict]] = None
    enum: List[str] = field(default_factory=list)
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    max_items: Optional[int] = None
    min_items: Optional[int] = None
    unique_items: Optional[bool] = None
    additional_properties: Union[bool, "SchemaProperties", None] = True
    nullable: Optional[bool] = None
    read_only: Optional[bool] = None
    write_only: Optional[bool] = None
    example: Optional[Union[str, int, float, bool, List, Dict]] = None
    examples: List[Optional[Union[str, int, float, bool, List, Dict]]] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return to_dict_helper(asdict(self))


@dataclass
class ParameterProperties:
    """Stores the properties of an OpenAPI parameter."""

    name: str = ""
    in_value: Optional[str] = None
    description: Optional[str] = None
    required: Optional[bool] = None
    deprecated: Optional[bool] = None
    allow_empty_value: Optional[bool] = None
    style: Optional[str] = None
    explode: Optional[bool] = None
    allow_reserved: Optional[bool] = None
    schema: Optional[SchemaProperties] = None
    example: Optional[Union[str, int, float, bool, List, Dict]] = None
    examples: List[Optional[Union[str, int, float, bool, List, Dict]]] = field(
        default_factory=list
    )
    content: Dict[str, SchemaProperties] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return to_dict_helper(asdict(self))


@dataclass
class OperationProperties:
    """Stores the properties of an OpenAPI operation."""

    operation_id: str = ""
    endpoint_path: str = ""
    http_method: str = ""
    summary: Optional[str] = None
    parameters: Dict[str, ParameterProperties] = field(default_factory=dict)
    request_body: Optional[Dict[str, SchemaProperties]] = None
    responses: Optional[Dict[str, ResponseProperties]] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_dict_helper(asdict(self))


@dataclass
class RequestData:
    """Encapsulates the request information derived from an operation."""

    endpoint_path: str
    http_method: str
    parameters: Dict[str, Any]
    request_body: Dict[str, Any]
    operation_properties: OperationProperties
    requirements: Optional["RequestRequirements"] = None


@dataclass
class RequestRequirements:
    """Captures dependencies required to satisfy a request edge."""

    edge: "OperationEdge"
    parameter_requirements: Dict[str, Any] = field(default_factory=dict)
    request_body_requirements: Dict[str, Any] = field(default_factory=dict)

    def generate_combinations(self) -> List["RequestRequirements"]:
        combinations: List[RequestRequirements] = []

        param_combinations: List[Dict[str, Any]] = []
        if self.parameter_requirements:
            for i in range(1, len(self.parameter_requirements) + 1):
                for subset in itertools.combinations(
                    self.parameter_requirements.keys(), i
                ):
                    param_combinations.append(
                        {key: self.parameter_requirements[key] for key in subset}
                    )

        body_combinations: List[Dict[str, Any]] = []
        if self.request_body_requirements:
            for i in range(1, len(self.request_body_requirements) + 1):
                for subset in itertools.combinations(
                    self.request_body_requirements.keys(), i
                ):
                    body_combinations.append(
                        {key: self.request_body_requirements[key] for key in subset}
                    )

        if not param_combinations and not body_combinations:
            return [self]

        if not param_combinations:
            for body_combination in body_combinations:
                combinations.append(
                    RequestRequirements(
                        edge=self.edge,
                        parameter_requirements={},
                        request_body_requirements=body_combination,
                    )
                )
            return combinations

        if not body_combinations:
            for param_combination in param_combinations:
                combinations.append(
                    RequestRequirements(
                        edge=self.edge,
                        parameter_requirements=param_combination,
                        request_body_requirements={},
                    )
                )
            return combinations

        for param_combination in param_combinations:
            for body_combination in body_combinations:
                combinations.append(
                    RequestRequirements(
                        edge=self.edge,
                        parameter_requirements=param_combination,
                        request_body_requirements=body_combination,
                    )
                )

        combinations.sort(
            key=lambda req: len(req.parameter_requirements)
            + len(req.request_body_requirements),
            reverse=True,
        )
        return combinations


@dataclass
class RequestResponse:
    """Pairs a request with its observed response."""

    request: RequestData
    response: requests.Response
    response_text: str


@dataclass
class ValueAction:
    """Represents an action taken by the value agent."""

    param_mappings: Optional[Dict[str, object]]
    body_mappings: Optional[Dict[str, object]]


@dataclass
class SimilarityValue:
    """Captures similarity metadata when comparing operations."""

    dependent_val: str = ""
    in_value: str = ""
    similarity: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response": self.dependent_val,
            "in_value": self.in_value,
            "similarity": self.similarity,
        }


__all__ = [
    "to_dict_helper",
    "ResponseProperties",
    "SchemaProperties",
    "ParameterProperties",
    "OperationProperties",
    "RequestData",
    "RequestRequirements",
    "RequestResponse",
    "ValueAction",
    "SimilarityValue",
]
