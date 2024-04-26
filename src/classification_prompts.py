FEW_SHOT_PARAMETER_CONSTRAINT_ERROR = '''
STATUS CODE: 400 
MESSAGE: Invalid parameter value
CLASSIFICATION: PARAMETER CONSTRAINT

STATUS CODE: 405 
MESSAGE: {"message": "One or more parameters are invalid", "details" : [{"field": "age", "constraint" : "must be a positive integer"}]} 
CLASSIFICATION: PARAMETER CONSTRAINT 

STATUS CODE: 400 
MESSAGE: <error><code>400</code><message>Age is not a valid parameter</message><details><field>age</field><constraint>must be a positive integer</constraint></details></error>
CLASSIFICATION: PARAMETER CONSTRAINT 
'''.strip()
FEW_SHOT_FORMAT_ERROR = '''
STATUS CODE: 500 
MESSAGE: Invalid parameter format
CLASSIFICATION: FORMAT

STATUS CODE: 406
MESSAGE: <error><code>400</code><message>Invalid request body format</message><details>The request body must be in XML format.</details></error>
CLASSIFICATION: FORMAT 

STATUS CODE: 422
MESSAGE: {"error": {"code": 422,"message": "Invalid parameter format","details": [{"field": "date","issue": "The date parameter must be in the format YYYY-MM-DD."}]}}
CLASSIFICATION: FORMAT
'''.strip()
FEW_SHOT_PARAMETER_DEPENDENCY_ERROR = '''
STATUS CODE: 412
MESSAGE: {"error": {"code": 412,"message": "Invalid parameter combination","details": [{"field": "paymentMethod","issue": "The 'creditCard' payment method requires the 'cardNumber' and 'expirationDate' parameters to be provided."}]}}
CLASSIFICATION: PARAMETER DEPENDENCY

STATUS CODE: 400 
MESSAGE: <error><code>400</code><message>Missing required parameters</message><details><field>shippingAddress</field><issue>The 'shippingAddress' parameter is required when the 'shipToAddress' parameter is set to true.</issue></details></error>
CLASSIFICATION: PARAMETER DEPENDENCY

STATUS CODE: 400 
MESSAGE: password field requires date 
CLASSIFICATION: PARAMETER DEPENDENCY 
'''.strip() 
FEW_SHOT_OPERATION_DEPENDENCY_ERROR = '''
STATUS CODE: 400 
MESSAGE: <error><code>400</code><message>Invalid operation sequence</message><details><operation>updateProfile</operation><issue>The 'updateProfile' operation cannot be performed until the user has completed the 'verifyEmail' operation.</issue></details></error>
CLASSIFICATION: OPERATION DEPENDENCY

STATUS CODE: 500 
MESSAGE: {"error": {"code": 400,"message": "Operation not allowed","details": [{"operation": "checkout","issue": "The 'checkout' operation cannot be performed until items have been added to the shopping cart."}]}}
CLASSIFICATION: OPERATION DEPENDENCY 

STATUS CODE: 424 
MESSAGE: User must log in first. 
CLASSIFICATION: OPERATION DEPENDENCY  
'''.strip()
CLASSIFICATION_PROMPT = '''
Given an error message, classify the error as one of "PARAMETER CONSTRAINT, FORMAT, PARAMETER DEPENDENCY, OPERATION DEPENDENCY". '''
CLASSIFICATION_SUFFIX = '''
Return the answer as simply "CLASSIFICATION: class". 
MESSAGE:
'''
FEW_SHOT_CLASSIFICATON_PREFIX = CLASSIFICATION_PROMPT + '\n' + FEW_SHOT_PARAMETER_DEPENDENCY_ERROR + '\n' + FEW_SHOT_FORMAT_ERROR + '\n' + FEW_SHOT_PARAMETER_CONSTRAINT_ERROR + '\n' + FEW_SHOT_OPERATION_DEPENDENCY_ERROR + '\n'

PARAMETER_CONSTRAINT_IDENTIFICATION_PREFIX = '''
Given a server response to an API call, and a list of parameters used to make that call, identify any parameters that need to be constrained, if any. For example:
MESSAGE: {"message": "One or more parameters are invalid", "details" : [{"field": "age", "constraint" : "must be a positive integer"}]}
PARAMETERS: age, name, date
IDENTIFICATION: age

MESSAGE: <error><code>400</code><message>Age is not a valid parameter</message><details><field>age</field><constraint>must be a positive integer</constraint></details></error>
PARAMETERS: age, name, date
IDENTIFICATION: age

MESSAGE: the email must be 10 characters long 
PARAMETERS: email, password, date
IDENTIFICATION: email

MESSAGE: invalid parameter values
PARAMETERS: variant, transaction_id, date
IDENTIFICATION: none

Return your answer in the format "IDENTIFICATION: " followed by a comma seperated list of identified parameters. If no parameters need to be constrained, return "IDENTIFICATION: none". Do not include any other information in your response. \n
'''
MESSAGE_HEADER = '''MESSAGE: '''
PARAMETERS_HEADER = '''PARAMETERS: '''
IDENTIFICATION_HEADER = '''IDENTIFICATION: '''

CONSTRAINT_EXTRACTION_PREFIX = '''
Given the server response message and parameter, return a JSON object to define the following dataclass to create a constrained schema for that parameter.

class SchemaProperties:
    """
    Class to store the properties of either the schema values, in the case of parameters, or the request body object values
    """
    type: Optional[str] = None
    format: Optional[str] = None
    description: Optional[str] = None
    items: 'SchemaProperties' = None
    properties: Dict[str, 'SchemaProperties'] = None # property name then schema content as value
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
    additional_properties: Union[bool, 'SchemaProperties', None] = True
    nullable: Optional[bool] = None
    read_only: Optional[bool] = None
    write_only: Optional[bool] = None
    example: Optional[Union[str, int, float, bool, List, Dict]] = None
    examples: List[Optional[Union[str, int, float, bool, List, Dict]]] = field(default_factory=list)

Ensure that the returned JSON follows the semantics of Schema Properties.
'''
EXAMPLE_GENERATION_PROMPT = '''
Given the server message, generate an example parameter value that would satisfy the server response for the provided paramter. 

MESSAGE: Field 'email' must be a valid email address.
PARAMETER: email
EXAMPLE: abc@gmail.com

MESSAGE: Field 'age' must be a positive integer.
PARAMETER: age
EXAMPLE: 25

Given the message and parameter, generate an example value. Return the example value as a string with format "EXAMPLE: <example value>". Do not include any additional information.
'''
EXTRACT_PARAMETER_DEPENDENCIES = '''
Given a server response message and a list of parameters, extract any parameter dependencies from the message, and return a list of tuples that highlights the parameter dependencies. 
Return parameter dependencies in the form of which parameters are required. Only include the response, and do not include any other information.
For example:

MESSAGE: The 'creditCard' payment method requires the 'cardNumber' and 'expirationDate' parameters to be provided.
PARAMETERS: paymentMethod, cardNumber, expirationDate, shippingAddress, shipToAddress
DEPENDENCIES: paymentMethod, cardNumber, expirationDate

MESSAGE: The 'shippingAddress' parameter is required when the 'shipToAddress' parameter is set to true.
PARAMETERS: shippingAddress, shipToAddress, password, date
DEPENDENCIES: shippingAddress, shipToAddress

MESSAGE: password field requires date
PARAMETERS: password, date
DEPENDENCIES: password, date

MESSAGE: The API requires parameters
PARAMETERS: transactionId, cardNumber, language
DEPENDENCIES: transcriptId, cardNumber, language
'''

DEFAULT_SYSTEM_MESSAGE = "You will receive instructions and examples. Your goal is to provide a systematic output using the knowledge from the instructions and examples provided."