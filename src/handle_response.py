from .classification_prompts import *
from .specification_parser import SchemaProperties
from bs4 import BeautifulSoup
from openai import OpenAI
import os
import json 

class ResponseHandler:
    def __init__(self):
        self.parser_type = "html.parser"
        self.language_model = ResponseLanguageModelHandler()
    def extract_response_text(self,response):
        if not response:
            raise ValueError()
        response_text = response.text
        result = ' '.join(BeautifulSoup(response_text, self.parser_type).stripped_strings)
        return result
    def classify_error(self, response):
        response_text = self.extract_response_text(response)
        return self.language_model.classify_response(response_text) 
    def handle_error(self, response, parameters):
        error_classification = self.classify_error(response)
        if error_classification == "PARAMETER CONSTRAINT":
            #identify parameter constraint and return new parameters and request body dictionary that specifies the parameters to use
            modified_parameter_schemas = self.language_model.extract_constrained_schemas(response, parameters)
            #merge schemas 
            for parameter in parameters:
                if parameter in modified_parameter_schemas:
                    parameters[parameter].schema = modified_parameter_schemas[parameter]
            return {"PARAMETER CONSTRAINT" : parameters}
        elif error_classification == "FORMAT":
            #should return map from parameter -> example
            parameter_format_examples = self.language_model.extract_parameter_formatting(response, parameters)
            for parameter in parameters:
                if parameter in parameter_format_examples:
                    parameters[parameter].example = parameter_format_examples[parameter]
            return {"FORMAT" : parameters}
        elif error_classification == "PARAMETER DEPENDENCY":
            required_parameters = self.language_model.extract_parameter_dependency(response, parameters)
            for parameter in parameters:
                if parameter in required_parameters:
                    parameters[parameter].required = True
            return {"PARAMETER DEPENDENCY": parameters}
        elif error_classification == "OPERATION DEPENDENCY":
            #return a list of operations that are dependent on each other
            return {"OPERATION DEPENDENCY": parameters}
        else:
            return None
    
class ResponseLanguageModelHandler:
    def __init__(self, language_model="OPENAI", **kwargs):
        if language_model == "OPENAI":
            env_var = os.getenv("OPENAI_API_KEY")
            self.language_model_engine = kwargs.get("language_model_engine", "gpt-4-turbo-preview")
            if env_var is None or env_var.strip() == "":
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
    def _extract_classification(self, response_text):
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
    def _extract_parameters_to_constrain(self, response_text, request_params):
        parameter_list = [parameter for parameter in request_params]
        #create a comma seperated string of parameters
        parameters = ",".join(parameter_list)
        parameters = "PARAMETERS: " + parameters + "\n"
        message = "MESSAGE: " + response_text + "\n"

        extracted_paramter_list = self._extract_constrained_parameter_list(self.language_model_query(PARAMETER_CONSTRAINT_IDENTIFICATION_PREFIX + message + parameters))
        return self._extract_constrained_parameter_list(extracted_paramter_list)
    def _generate_parameter_value(self, parameters_to_generate_for, response_text):
        example_value_map = {}
        for parameter in parameters_to_generate_for:
            raw_result = self.language_model_query(EXAMPLE_GENERATION_PROMPT + MESSAGE_HEADER + response_text + PARAMETERS_HEADER + parameter)
            #parse the result to get the example value
            example_value = raw_result.split("EXAMPLE:")[1].strip()
            example_value_map[parameter] = example_value
        return example_value_map
    def define_constrained_schema(self, parameter, response_text):
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
    def extract_constrained_schemas(self, response_text, request_params):
        parameters_to_constrain = self._extract_constrained_parameter_list(self._extract_parameters_to_constrain(response_text, request_params))
        constrained_schemas = {}
        for parameter in parameters_to_constrain: 
            if parameter in request_params:
                parameter_schema = request_params[parameter].schema
                constrained_schemas[parameter] = self.define_constrained_schema(parameter, parameter_schema, response_text)
        return constrained_schemas
    def classify_response(self, response_text):
        return self._extract_classification(self.language_model_query(FEW_SHOT_CLASSIFICATON_PREFIX + response_text + CLASSIFICATION_SUFFIX))
    def extract_parameter_formatting(self, response_text, request_params):
        params_list = self._extract_constrained_parameter_list(self._extract_parameters_to_constrain(response_text, request_params))
        return self._generate_parameter_value(params_list, response_text)
    def extract_parameter_dependency(self, response_text, request_params):
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
