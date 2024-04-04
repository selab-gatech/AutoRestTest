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
