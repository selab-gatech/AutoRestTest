from .classification_prompts import *
from bs4 import BeautifulSoup
from openai import OpenAI
import os
from dataclasses import dataclass


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
    def handle_error(self, response):
        error_classification = self.classify_error(response)
        if error_classification == "PARAMETER CONSTRAINT":
            #identify parameter constraint and return new parameters and request body dictionary that specifies the parameters to use
            modified_parameter_schemas = self.language_model.extract_parameter_constraints(response)
            return modified_parameter_schemas
        elif error_classification == "FORMAT":
            pass
        elif error_classification == "PARAMETER DEPENDENCY":
            pass
        elif error_classification == "OPERATION DEPENDENCY":
            pass
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
    def language_model_query(self,query):
        #get openai chat completion 
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
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
    def define_constrained_schema(self, parameter, parameter_schema, response_text):
        pass
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
