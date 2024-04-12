from typing import Dict, TYPE_CHECKING

from requests import Response

from .classification_prompts import *
from bs4 import BeautifulSoup
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv() # load environmental vars for OpenAI API key

if TYPE_CHECKING:
    from .generate_graph import OperationNode
    from .specification_parser import SchemaProperties, ParameterProperties
    from src.request_generator import NaiveRequestGenerator, RequestData


class ResponseHandler:
    def __init__(self):
        self.parser_type = "html.parser"
        self.language_model = ResponseLanguageModelHandler("OPENAI", os.getenv("OPENAI_API_KEY"))

    def extract_response_text(self, response: Response):
        if not response:
            raise ValueError()
        response_text = response.text
        result = ' '.join(BeautifulSoup(response_text, self.parser_type).stripped_strings) # process HTML response
        return result

    def classify_error(self, response: Response):
        response_text = self.extract_response_text(response)
        return self.language_model.classify_response(response_text) 
    
    def is_valid_dependency(self, failed_response: Response, tentative_response: Response):
        if failed_response is None or tentative_response is None:
            return False
        if failed_response.status_code // 100 == 2:
            return True
        return False

    def test_tentative_edge(self, request_generator: 'NaiveRequestGenerator', failed_operation_node: 'OperationNode', tentative_edge):
        tentative_operation_node = tentative_edge.destination
        tentative_response = request_generator.create_and_send_request(tentative_operation_node)
        failed_response = request_generator.create_and_send_request(failed_operation_node)
        if tentative_response is not None and failed_response is not None and self.is_valid_dependency(failed_response, tentative_response):
            return True
        return False
    
    def handle_operation_dependency_error(self, request_generator: 'NaiveRequestGenerator', failed_operation_node: 'OperationNode'):
        '''
        Handle the operation dependency error by trying tentative edges.
        '''
        if not failed_operation_node.tentative_edges:
            return
        sorted_edges = sorted(failed_operation_node.tentative_edges, key=lambda x: list(x.similar_parameters.values())[0].similarity, reverse=True) # sort tentative edges by their one parameter similarity value
        for tentative_edge in sorted_edges:
            # Send a request to the tentative operation and check the response
            if self.test_tentative_edge(request_generator, failed_operation_node, tentative_edge):
                request_generator.operation_graph.add_operation_edge(
                    failed_operation_node.operation_id,
                    tentative_edge.destination.operation_id,
                    tentative_edge.similar_parameters
                )
                failed_operation_node.tentative_edges.remove(tentative_edge)
                print(
                    f"Updated the graph with a new edge from {failed_operation_node.operation_id} to {tentative_edge.destination.operation_id}")
                return
        # add highest similarity edge
        request_generator.operation_graph.add_operation_edge(
            failed_operation_node.operation_id,
            sorted_edges[0].destination.operation_id,
            sorted_edges[0].similar_parameters
        )
        print(
            f"Updated the graph with a new edge from {failed_operation_node.operation_id} to {sorted_edges[0].destination.operation_id}")
        failed_operation_node.tentative_edges.remove(sorted_edges[0])

    def handle_parameter_constraint_error(self, response_text: str, parameters: Dict[str, 'SchemaProperties']):
        modified_parameter_schemas = self.language_model.extract_constrained_schemas(response_text, parameters)
        for parameter in parameters:
            if parameter in modified_parameter_schemas:
                print(f"Updating parameter {parameter} with new schema")
                parameters[parameter] = modified_parameter_schemas[parameter]

    def handle_format_constraint_error(self, response_text: str, parameters: Dict[str, 'SchemaProperties']):
        parameter_format_examples = self.language_model.extract_parameter_formatting(response_text, parameters)
        for parameter in parameters:
            if parameter in parameter_format_examples:
                print(f"Updating parameter {parameter} with new example value")
                parameters[parameter].example = parameter_format_examples[parameter]

    def handle_parameter_dependency_error(self, response_text: str, parameters: Dict[str, 'SchemaProperties']):
        required_parameters = self.language_model.extract_parameter_dependency(response_text, parameters)
        for parameter in parameters:
            if parameter in required_parameters:
                print(f"Updating parameter {parameter} to required")
                parameters[parameter].required = True

    def handle_error(self, response: Response, operation_node: 'OperationNode', request_data: 'RequestData', request_generator: 'NaiveRequestGenerator'):
        # TODO: Implement differentiation of parameters vs req body
        error_classification = self.classify_error(response)
        query_parameters: Dict[str, 'ParameterProperties'] = request_data.operation_properties.parameters

        request_body: Dict[str, 'SchemaProperties'] = request_data.operation_properties.request_body
        simplified_parameters: Dict[str, 'SchemaProperties'] = {parameter: properties.schema for parameter, properties in query_parameters.items()}
        response_text = self.extract_response_text(response)

        # REMINDER: parameters = Dict[str, ParameterProperties] -> schema = SchemaProperties
        # REMINDER: request_body = Dict[str, SchemaProperties]
        if error_classification == "PARAMETER CONSTRAINT":
            #identify parameter constraint and return new parameters and request body dictionary that specifies the parameters to use
            self.handle_parameter_constraint_error(response_text, simplified_parameters)
            self.handle_parameter_constraint_error(response_text, request_body)
        elif error_classification == "FORMAT":
            #should return map from parameter -> example
            self.handle_format_constraint_error(response_text, simplified_parameters)
            self.handle_format_constraint_error(response_text, request_body)
        elif error_classification == "PARAMETER DEPENDENCY":
            self.handle_parameter_dependency_error(response_text, simplified_parameters)
            self.handle_parameter_dependency_error(response_text, request_body)
        elif error_classification == "OPERATION DEPENDENCY":
            self.handle_operation_dependency_error(request_generator, operation_node)
        else:
            return None
    
class ResponseLanguageModelHandler:
    def __init__(self, language_model="OPENAI", api_key = None, **kwargs):
        if language_model == "OPENAI":
            self.api_key = api_key
            self.language_model_engine = kwargs.get("language_model_engine", "gpt-4-turbo-preview")
            if api_key is None or api_key.strip() == "":
                raise ValueError()
            self.client = OpenAI()
        else:
            raise Exception("Unsupported language model")

    def language_model_query(self,query, json_mode=False):
        #get openai chat completion 
        if json_mode:
            response = self.client.chat.completions.create(
                model=self.language_model_engine, 
                messages = [
                    {'role': 'user', 'content' : query}
                ], 
                response_format={ "type": "json_object" }
            )
        else: 
            response = self.client.chat.completions.create(
                model=self.language_model_engine,
                messages=[
                    {'role': 'user', 'content': query},
                ]
            )
        return response.choices[0].message.content.strip()

    def _extract_classification(self, response_text: str):
        classification = None
        if response_text is None: 
            return classification 
        if "PARAMETER CONSTRAINT" in response_text:
            classification = "PARAMETER CONSTRAINT"
        elif "FORMAT" in response_text:
            classification = "FORMAT"
        elif "PARAMETER DEPENDENCY" in response_text:
            classification = "PARAMETER DEPENDENCY"
        elif "OPERATION DEPENDENCY" in response_text:
            classification = "OPERATION DEPENDENCY"
        return classification

    def _extract_constrained_parameter_list(self, language_model_response):
        if "IDENTIFICATION:" not in language_model_response:
            return None
        elif language_model_response.strip() == 'IDENTIFICATION:' or language_model_response.strip() == 'IDENTIFICATION: none':
            return None
        else:
            return language_model_response.split("IDENTIFICATION:")[1].strip().split(",")

    def _extract_parameters_to_constrain(self, response_text: str, request_params):
        parameter_list = [parameter for parameter in request_params]
        #create a comma seperated string of parameters
        parameters = ",".join(parameter_list)
        parameters = "PARAMETERS: " + parameters + "\n"
        message = "MESSAGE: " + response_text + "\n"

        extracted_parameter_list = self._extract_constrained_parameter_list(self.language_model_query(PARAMETER_CONSTRAINT_IDENTIFICATION_PREFIX + message + parameters))
        #return self._extract_constrained_parameter_list(extracted_paramter_list)
        return extracted_parameter_list

    def _generate_parameter_value(self, parameters_to_generate_for, response_text: str):
        example_value_map = {}
        for parameter in parameters_to_generate_for:
            raw_result = self.language_model_query(EXAMPLE_GENERATION_PROMPT + MESSAGE_HEADER + response_text + PARAMETERS_HEADER + parameter)
            #parse the result to get the example value
            example_value = raw_result.split("EXAMPLE:")[1].strip()
            example_value_map[parameter] = example_value
        return example_value_map

    def define_constrained_schema(self, parameter, response_text: str):
        extract_query = CONSTRAINT_EXTRACTION_PREFIX + MESSAGE_HEADER + response_text + PARAMETERS_HEADER + parameter
        json_schema_properties = self.language_model_query(extract_query, json_mode=True)
        #read the json string into a dictionary 
        schema_properties = json.loads(json_schema_properties)
        #map it to a schema properties dataclass, ensure checking of failures and only map what is possible
        schema = SchemaProperties()
        for key, value in schema_properties.items():
            if hasattr(schema, key):
                setattr(schema, key, value)
        return schema

    def extract_constrained_schemas(self, response_text: str, request_params):
        """
        Find the parameters that need to be constrained, then processes constraints
        :param response_text:
        :param request_params:
        :return:
        """
        parameters_to_constrain = self._extract_constrained_parameter_list(self._extract_parameters_to_constrain(response_text, request_params))
        constrained_schemas = {}
        for parameter in parameters_to_constrain: 
            if parameter in request_params:
                parameter_schema = request_params[parameter].schema
                constrained_schemas[parameter] = self.define_constrained_schema(parameter, response_text)
        return constrained_schemas

    def classify_response(self, response_text: str):
        return self._extract_classification(self.language_model_query(FEW_SHOT_CLASSIFICATON_PREFIX + response_text + CLASSIFICATION_SUFFIX))

    def extract_parameter_formatting(self, response_text: str, request_params):
        params_list = self._extract_constrained_parameter_list(self._extract_parameters_to_constrain(response_text, request_params))
        return self._generate_parameter_value(params_list, response_text)
    
    def extract_parameter_dependency(self, response_text: str, request_params):
        list_of_request_parameters = [parameter for parameter in request_params]
        parameters = ",".join(list_of_request_parameters)
        parameters = "PARAMETERS: " + parameters + "\n"
        message = "MESSAGE: " + response_text + "\n"
        extracted_paramter_list = self._extract_constrained_parameter_list(self.language_model_query(EXTRACT_PARAMETER_DEPENDENCIES + message + parameters)) 
        #clean the response 
        if extracted_paramter_list is None:
            return None
        else:
            return set(extracted_paramter_list.split("DEPENDENCIES: ")[1].strip().split(","))

class DummyRequest:
    def __init__(self):
        self.response = 'if email is provided age must be set'

class DummySchema :
    def __init__(self):
        pass