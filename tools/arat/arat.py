import re
import os
import sys
import copy
import json
import time
import math
import rstr
import base64
import random
import string
import prance
import difflib
import requests
import datetime
import functools
from collections import defaultdict


def generate_object(object_definition, operation):
    random_object = {}
    properties = list(object_definition.items())
    num_fields = random.randint(0, len(properties))
    selected_properties = random.sample(properties, num_fields)
    for prop_element in selected_properties:
        prop, prop_def = prop_element
        random_object[prop] = get_next_parameter_value(operation, prop_def)
    return random_object


def generate_random_string_from_pattern(pattern, min_length=0, max_length=None):
    try:
        generated_str = rstr.xeger(pattern)
        if max_length:
            generated_str = generated_str[min_length:max_length-1]
        return generated_str
    except Exception:
        pass
    return None


def get_value(param_type, operation=None, parameter=None, object_definition=None, param_format=None, array_item_type=None, response_values=None):

    if random.random() < 0.1:
        param_type = random.choice(['string', 'integer', 'number', 'boolean'])
    min = 0
    max = None
    pattern = None
    if parameter:
        if "minLength" in parameter:
            min = parameter["minLength"]
        if "maxLength" in parameter:
            max = parameter["maxLength"]
        if "pattern" in parameter:
            pattern = parameter["pattern"]
    if param_type == 'string':
        if param_format is None:
            param_format = random.choice(['date', 'date-time', 'password', 'byte', 'binary'])
        value = None

        if pattern and random.random() < 0.9:
            value = generate_random_string_from_pattern(pattern, min, max)
        if value:
            return value
        else:
            if param_format == 'date':
                random_date = datetime.date.fromtimestamp(random.randint(0, int(datetime.datetime.now().timestamp())))
                return random_date.strftime('%Y-%m-%d')
            elif param_format == 'date-time':
                random_datetime = datetime.datetime.fromtimestamp(
                    random.randint(0, int(datetime.datetime.now().timestamp())))
                return random_datetime.strftime('%Y-%m-%dT%H:%M:%SZ')
            elif param_format == 'password':
                random_password_length = random.randint(5, 10)
                characters = string.ascii_letters + string.digits + string.punctuation
                return ''.join(random.choice(characters) for _ in range(random_password_length))
            elif param_format == 'byte':
                random_byte_length = random.randint(1, 10)
                return base64.b64encode(os.urandom(random_byte_length)).decode('utf-8')
            elif param_format == 'binary':
                random_binary_length = random.randint(1, 10)
                return ''.join(random.choice(['0', '1']) for _ in range(random_binary_length))
    elif param_type == 'integer':
        return random.randint(-10000, 10000)
    elif param_type == 'number':
        return random.uniform(-10000, 10000)
    elif param_type == 'boolean':
        return random.choice([True, False])
    elif param_type == 'object':
        if object_definition:
            nested_object = {}
            num_properties_to_select = random.randint(1, len(object_definition))
            selected_properties = random.sample(list(object_definition.items()), num_properties_to_select)

            for prop, prop_def in selected_properties:
                nested_prop_type = prop_def.get('type', None)
                nested_object[prop] = get_value(nested_prop_type, operation=operation, parameter=prop_def,
                                                object_definition=prop_def.get('properties', None),
                                                response_values=response_values)

            return nested_object
        else:
            return {}
    elif param_type == 'array':
        array_length = 1
        if array_item_type == 'object' and object_definition:
            return [generate_object(object_definition, operation=operation)]
        else:
            return [get_value(array_item_type, operation=operation) for _ in range(array_length)]
    else:
        return None

@functools.lru_cache(maxsize=1024)
def get_random_values_from_description(description):
    all_values = list(set(re.findall(r"[\w,]+", description) + re.findall(r"'([^']+)'", description) + re.findall(r"`([^`]+)`", description)+ re.findall(r'"([^"]+)"', description)))

    if all_values:
        return random.choice(all_values)
    else:
        return [None]

def is_value_of_type(value, param_type):
    if param_type == 'integer' and isinstance(value, int):
        return True
    elif param_type == 'number' and isinstance(value, float):
        return True
    elif param_type == 'string' and isinstance(value, str):
        return True
    elif param_type == 'boolean' and isinstance(value, bool):
        return True
    elif param_type == "array" and isinstance(value, list):
        return True
    elif param_type == "object" and isinstance(value, dict):
        return True
    else:
        return False

def extract_response_values(response, op):
    try:
        if isinstance(response, list):
            val = random.choice(response)
            extract_response_values(val, op)
        elif isinstance(response, dict):
            key, value = random.choice(list(response.items()))
            if key in response_values:
                if value not in response_values[key]:
                    response_values[key].append(value)
            if isinstance(value, dict) or isinstance(value, list):
                extract_response_values(value, op)
            else:
                if key not in response_values:
                    response_values[key] = []
                if value not in response_values[key]:
                    if key not in producer:
                        producer[key] = []
                    if op["operation_id"] not in producer[key]:
                        if op["method"] == "get" and len(producer[key]) > 0:
                            pass
                        else:
                            producer[key].append(op["operation_id"])
                    response_values[key].append(value)
    except Exception as e:
        pass

def generate_parameter_values(operations):
    generated_values = {}

    for operation in operations:
        operation_id = operation['operation_id']
        generated_values[operation_id] = []

        for parameter in operation['parameters']:
            param_name = parameter['name']

            value = get_next_parameter_value(operation, parameter)

            if value is not None:
                generated_values[operation_id].append({param_name: value})

    return generated_values

def execute_operations(base_url, selected_operation, selected_parameters):
    method, path = selected_operation['method'], selected_operation['path']
    query_params, body_params = {}, {}
    media_types = selected_operation.get('consumes', [
        'application/json', 'application/x-www-form-urlencoded'
    ])

    def send_request(content_type):
        headers = {"Content-Type": content_type}

        try:
            if content_type == 'application/x-www-form-urlencoded':
                try:
                    if method == 'get':
                        return requests.get(url, params=query_params, headers=headers, data=body_params)
                    elif method == 'post':
                        return requests.post(url, params=query_params, headers=headers, data=body_params)
                    elif method == 'put':
                        return requests.put(url, params=query_params, headers=headers, data=body_params)
                    elif method == 'delete':
                        return requests.delete(url, params=query_params, headers=headers, data=body_params)
                    elif method == 'patch':
                        return requests.patch(url, params=query_params, headers=headers, data=body_params)
                    elif method == 'head':
                        return requests.head(url, headers=headers, params=query_params)
                except Exception:
                    if method == 'get':
                        return requests.get(url, params=query_params, headers=headers, json=body_params)
                    elif method == 'post':
                        return requests.post(url, params=query_params, headers=headers, json=body_params)
                    elif method == 'put':
                        return requests.put(url, params=query_params, headers=headers, json=body_params)
                    elif method == 'delete':
                        return requests.delete(url, params=query_params, headers=headers, json=body_params)
                    elif method == 'patch':
                        return requests.patch(url, params=query_params, headers=headers, json=body_params)
                    elif method == 'head':
                        return requests.head(url, headers=headers, params=query_params)
            else:
                if method == 'get':
                    return requests.get(url, params=query_params, headers=headers, json=body_params)
                elif method == 'post':
                    return requests.post(url, params=query_params, headers=headers, json=body_params)
                elif method == 'put':
                    return requests.put(url, params=query_params, headers=headers, json=body_params)
                elif method == 'delete':
                    return requests.delete(url, params=query_params, headers=headers, json=body_params)
                elif method == 'patch':
                    return requests.patch(url, params=query_params, headers=headers, json=body_params)
                elif method == 'head':
                    return requests.head(url, headers=headers, params=query_params)
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            return None

    for param_value_dict in selected_parameters:
        for param_name, param_value in param_value_dict.items():
            param = next((p for p in selected_operation['parameters'] if p['name'] == param_name), None)
            if param is None: continue
            if "schema" in param:
                param_location, param_type = param['in'], param["schema"].get('type')
            else:
                param_location, param_type = param['in'], param.get('type')
            if param_location == 'path':
                path = path.replace(f'{{{param_name}}}', str(param_value))
            elif param_location == 'query':
                query_params[param_name] = param_value
            elif isinstance(param_value, list):
                body_params = param_value
            elif isinstance(param_value, dict):
                for tt in param_value:
                    body_params[tt] = param_value[tt]
            else:
                if isinstance(body_params, dict):
                    body_params[param_name] = param_value if param_type != 'array' else (param_value if isinstance(param_value, list) else [param_value])
    url = base_url + path
    if selected_operation['operation_id'] in cached_media_type:
        response = send_request(cached_media_type[selected_operation['operation_id']])
    else:
        for media_type in media_types:
            response = send_request(media_type)
            if response and 200 <= response.status_code < 300:
                cached_media_type[selected_operation['operation_id']] = media_type
                break

    return response, path, query_params, body_params


def get_mutated_value(param_type):
    # Get a list of all possible types
    all_types = ['string', 'integer', 'number', 'boolean', 'object', 'array']

    # Remove the original param_type from the list
    all_types.remove(param_type)

    # Randomly choose a new type different from the original param_type
    mutated_type = random.choice(all_types)

    # Generate a value with the new mutated type
    return get_value(mutated_type)

def perform_parameter_mutation(selected_parameters, selected_operation):
    mutated_parameters = []

    # Mutate media type
    if random.uniform(0, 1) < MUTATION_RATE:
        media_types = selected_operation.get('consumes', [
            'application/json', 'application/xml', 'application/x-www-form-urlencoded',
            'multipart/form-data', 'text/plain; charset=utf-8', 'text/html',
            'application/pdf', 'image/png'
        ])
        new_media_type = random.choice(media_types)
        selected_operation['consumes'] = [new_media_type]
    elif selected_operation['operation_id'] in cached_media_type:
        selected_operation['consumes'] = cached_media_type[selected_operation['operation_id']]

    def mutate_http_method(method):
        all_methods = ['get', 'post', 'put', 'delete', 'patch']
        allowed_methods = [m for m in all_methods if m != method]
        return random.choice(allowed_methods)

    method = selected_operation['method']

    if random.uniform(0, 1) < MUTATION_RATE:
        selected_operation['method'] = mutate_http_method(method)


    for param_value_dict in selected_parameters:
        for param_name, param_value in param_value_dict.items():
            param = None

            # Find the parameter
            for parameter in selected_operation['parameters']:
                if parameter['name'] == param_name:
                    param = parameter
                    break

            # Mutate "parameter type" randomly
            if random.uniform(0, 1) < MUTATION_RATE:
                if "schema" in param:
                    mutated_value = get_mutated_value(param["schema"]['type'])
                else:
                    mutated_value = get_mutated_value(param['type'])
                if mutated_value is not None:
                    mutated_parameters.append({param_name: mutated_value})
                else:
                    mutated_parameters.append(param_value_dict)
            else:
                mutated_parameters.append(param_value_dict)

    return mutated_parameters, selected_operation

def analyze_information(spec):
    operations = []
    parameters_frequency = defaultdict(int)

    for path, path_data in spec['paths'].items():
        for method, operation_data in path_data.items():
            if method in ['get', 'post', 'put', 'delete', 'patch']:
                operation_id = operation_data['operationId']
                operations.append({
                    'operation_id': operation_id,
                    'method': method,
                    'path': path,
                    'parameters': operation_data.get('parameters', []),
                    'responses': operation_data.get('responses', {})
                })

                for parameter in operation_data.get('parameters', []):
                    param_name = parameter['name']
                    parameters_frequency[param_name] += 1

                for response_code, response_data in operation_data.get('responses', {}).items():
                    schema = response_data.get('schema', {}).get('properties', {})
                    for response_property in schema.keys():
                        if response_property in parameters_frequency:
                            parameters_frequency[response_property] += 1

    return operations, parameters_frequency


def initialize_q_learning(operations, parameters_frequency):
    alpha = 0.1  # Learning rate
    gamma = 0.99  # Discount factor
      # Exploration rate

    # Initialize Q-value using parameter frequency
    q_table = {}
    for operation in operations:
        operation_id = operation['operation_id']
        q_table[operation_id] = {}
        q_value[operation_id] = {}
        q_value[operation_id]["response"] = 0
        q_value[operation_id]["request"] = 0
        q_value[operation_id]["random"] = 0
        q_value[operation_id]["specification"] = 0
        q_value[operation_id]["default"] = 0
        for parameter in operation['parameters']:
            param_name = parameter['name']
            q_table[operation_id][param_name] = parameters_frequency[param_name]

    return alpha, gamma, q_table


import json


def report_http_500_errors():
    # Prepare the report content
    report_content = "Total number of Found 500 Errors: " + str(total_n[0]) + "\n"
    report_content += "This report all errors that has different set of parameters, so we recommend to further explore them.\n"
    report_content += "HTTP 500 Error Report:\n"
    json_report = json.dumps(http_500_details, indent=4)
    report_content += json_report

    # Specify the filename
    filename = 'http_500_error_report.txt'

    # Open the file in write mode and write the report content
    with open(filename, 'w') as file:
        file.write(report_content)

    print(f"Report saved to {filename}")


def update_q_table(q_table, alpha, gamma, selected_operation, selected_parameters, response, path, query_params, body_params):
    operation_id = selected_operation['operation_id']
    if response is None:
        reward = -10
        q_value[operation_id][ss[0]] = q_value[operation_id][ss[0]] - 1
    if response.status_code == 401:
        reward = -1
    elif 200 <= response.status_code < 300:
        q_value[operation_id][ss[0]] = q_value[operation_id][ss[0]] + 1
        reward = -1
    elif 400 <= response.status_code:
        q_value[operation_id][ss[0]] = q_value[operation_id][ss[0]] - 1
        reward = 1
    else:
        q_value[operation_id][ss[0]] = q_value[operation_id][ss[0]] - 1
        reward = -5

    if response.status_code == 500:
        if operation_id not in http_500_details:
            http_500_details[operation_id] = {}
        if selected_operation['path'] not in http_500_details[operation_id]:
            http_500_details[operation_id][selected_operation['path']] = {}
        keys_list = list(selected_parameters.keys())
        sorted_keys_list = sorted(keys_list.keys())
        concatenated_keys = '_'.join(sorted_keys_list)
        if concatenated_keys not in http_500_details[operation_id][selected_operation['path']]:
            http_500_details[operation_id][selected_operation['path']][concatenated_keys] = []
            error_details = {
                "path": path,
                "query_parameters": query_params,
                "body_parameters": body_params,
                "response": response
            }
            http_500_details[operation_id].append(error_details)
            total_n[0] = total_n[0] + 1

    for param_value_dict in selected_parameters:
        for param_name, param_value in param_value_dict.items():
            if reward == -1:
                if param_name not in previous_request:
                    previous_request[param_name] = []
                if selected_operation['method'] == "post" or selected_operation['method'] == "get":
                    if param_value not in previous_request[param_name]:
                        previous_request[param_name].append(param_value)
                    if param_name not in producer:
                        producer[param_name] = []
                    if operation_id not in producer[param_name]:
                        if (selected_operation['method'] == "get" and len(producer[param_name]) > 0):
                            pass
                        else:
                            producer[param_name].append(operation_id)
                    if param_name not in consumer:
                        consumer[param_name] = []
                    if operation_id not in consumer[param_name]:
                        consumer[param_name].append(operation_id)

                else:
                    if operation_id not in consumer:
                        consumer[operation_id] = []
                    if param_name not in consumer[operation_id]:
                        consumer[operation_id].append(param_name)
                    for k in range(len(previous_request[param_name])):
                        if previous_request[param_name][k] == param_value:
                            del(previous_request[param_name][k])
                            break

            old_q_value = q_table[operation_id][param_name]
            max_q_value_next_state = max(q_table[operation_id].values())
            new_q_value = old_q_value + alpha * (reward + gamma * max_q_value_next_state - old_q_value)
            q_table[operation_id][param_name] = new_q_value

def adapt_testing_strategy(iteration, max_iterations_without_improvement):
    if iteration % max_iterations_without_improvement == 0:
        EPSILON[0] = min(1, EPSILON[0] * 1.1)

def select_operations_and_parameters(operations, parameter_values, q_table):
    method_priority = {'post': 5}

    sorted_operations = sorted(operations, key=lambda op: (
        (sum(q_table[op['operation_id']].values()) / len(q_table[op['operation_id']])) if len(
            q_table[op['operation_id']]) > 0 else 0,
        method_priority.get(op['method'], 0)), reverse=True)

    if random.uniform(0, 1) < EPSILON[0]:
        # Exploration: Choose a random operation
        selected_operation = random.choice(sorted_operations)
    else:
        # Exploitation: Choose the operation with the best Q-value
        selected_operation = sorted_operations[0]

    operation_id = selected_operation['operation_id']
    all_parameters = parameter_values[operation_id]
    required_parameters = [param for param in all_parameters if any(param_name in param for param_name in [param_data['name'] for param_data in selected_operation['parameters'] if param_data.get('required', False)])]
    optional_parameters = [param for param in all_parameters if param not in required_parameters]

    if random.uniform(0, 1) < EPSILON[0]:
        num_random_parameters = random.randint(0, len(optional_parameters))
        selected_parameters = required_parameters + random.sample(optional_parameters, num_random_parameters)
    else:
        selected_parameters = required_parameters
        sorted_optional_parameters = sorted(
            optional_parameters,
            key=lambda param: max(q_table[operation_id][param_name] for param_name in param),
            reverse=True
        )

        num_optional_parameters = random.randint(0, len(sorted_optional_parameters))
        selected_parameters += sorted_optional_parameters[:num_optional_parameters]

    return selected_operation, selected_parameters

def get_next_parameter_value(operation, parameter):
    if "name" not in parameter:
        param_name="MKobject"
    else:
        param_name = parameter['name']

    param_format = None
    object_schema = None
    array_item_type = None
    if "schema" in parameter:
        param_type = parameter["schema"]['type']
        if "format" in parameter["schema"]:
            param_format = parameter["schema"]['format']
        if "properties" in parameter["schema"]:
            object_schema = parameter["schema"]["properties"]
        if param_type == 'array' and "items" in parameter["schema"]:
            array_item_type = parameter["schema"]["items"]["type"]
            if "properties" in parameter["schema"]["items"]:
                object_schema = parameter["schema"]["items"]["properties"]
    else:
        param_type = parameter['type']
        if param_type == 'array' and "items" in parameter:
            array_item_type = parameter["items"]["type"]
            if "properties" in parameter["items"]:
                object_schema = parameter["items"]["properties"]
        if "format" in parameter:
            param_format = parameter['format']

    def random_value_from_dict(data_dict):
        if data_dict:
            key, value = random.choice(list(data_dict.items()))
            if value:
                return random.choice(value)

        return None

    def default_values(p):
        default_values = {
            'string': 'string',
            'integer': 1,
            'number': 1.1,
            'boolean': True,
            'array': [],
            'object': {}
        }
        default_values_formats = {
            'string': {
                'date-time': '1970-01-01T00:00:00Z'
            }
        }

        if param_type in default_values:
            if param_format and param_type in default_values_formats and param_format in default_values_formats[
                param_type]:
                return default_values_formats[param_type][param_format]
            else:
                return default_values[param_type]

        return None

    def response(p):
        max_similarity = -1
        most_similar_key = None

        for key in response_values:
            similarity = difflib.SequenceMatcher(None, param_name, key).ratio()
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_key = key

        if most_similar_key is not None and response_values[most_similar_key]:
            return random.choice(response_values[most_similar_key])
        else:
            return None

    def request(p):
        max_similarity = -1
        most_similar_key = None

        for key in previous_request:
            similarity = difflib.SequenceMatcher(None, param_name, key).ratio()
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_key = key
        if most_similar_key is not None and previous_request[most_similar_key]:
            return random.choice(previous_request[most_similar_key])
        else:
            return None

    def spec(p):
        value_candidates = []
        if 'enum' in p:
            value_candidates.extend(p['enum'])
        if 'example' in p:
            value_candidates.append(p['example'])
        if 'description' in p:
            value_candidates.append(get_random_values_from_description(p['description']))

        if value_candidates:
            return random.choice(value_candidates)
        else:
            return None

    sources = [
        ('specification', spec),
        ('request', lambda p: random_value_from_dict(previous_request) if random.random() < 0.1 else request(p)),
        ('response', lambda p: random_value_from_dict(response_values) if random.random() < 0.1 else response(p)),
        ('random', lambda p: get_value(param_type, param_format=param_format, object_definition=object_schema,
                                       array_item_type=array_item_type, operation=operation)),
        ('default', default_values)
    ]

    def q_value_based_choice(p):

        source_weights = {
            'specification': q_value[operation['operation_id']].get('specification'),
            'request': q_value[operation['operation_id']].get('request'),
            'response': q_value[operation['operation_id']].get('response'),
            'random': q_value[operation['operation_id']].get('random'),
            'default': q_value[operation['operation_id']].get('default')
        }

        selected_source = max(source_weights, key=source_weights.get)

        source_func = {
            'specification': spec,
            'request': lambda p: random_value_from_dict(previous_request) if random.random() < 0.1 else request(p),
            'response': lambda p: random_value_from_dict(response_values) if random.random() < 0.1 else response(p),
            'random': lambda p: get_value(param_type, param_format=param_format, object_definition=object_schema,
                                          array_item_type=array_item_type, operation=operation),
            'default': default_values
        }
        ss[0] = selected_source
        return source_func[selected_source](p)

    if random.uniform(0, 1) < EPSILON[0]:
        random.shuffle(sources)  # Randomi`ze the order of the sources
        value = None
        if random.uniform(0, 1) < EPSILON[0]:
            # Exploration: Choose a random source
            random_source = random.choice(sources)
            value = random_source[1](parameter)
        else:
            # Exploitation: Use the sources in the shuffled order
            for source_name, source_func in sources:
                value = source_func(parameter)
                if array_item_type and value and not isinstance(value, list):
                    value = [value]
                if value is not None and is_value_of_type(value, param_type):
                    break
    else:
        # Exploitation: Choose the source based on Q-value
        value = q_value_based_choice(parameter)
    # If no value is found from the sources above, return the default value
    return value if value is not None and is_value_of_type(value, param_type) else get_value(param_type, param_format=param_format, object_definition=object_schema, array_item_type=array_item_type, operation=operation)


def main():
    # Read Specification
    openapi_spec_file = sys.argv[1]
    openapi_spec = prance.ResolvingParser(openapi_spec_file).specification
    operations, parameters_frequency = analyze_information(openapi_spec)
    alpha, gamma, q_table = initialize_q_learning(operations, parameters_frequency)

    start_time = time.time()
    time_limit = sys.argv[3]
    iteration = 0
    max_iterations_without_improvement = 10

    while True:
        elapsed_time = time.time() - start_time
        if elapsed_time >= time_limit:
            break
        parameter_values = generate_parameter_values(operations)
        selected_operation, selected_parameters = select_operations_and_parameters(operations, parameter_values,
                                                                                   q_table)

        # Run all producer operations if the selected_operation is a consumer operation
        if selected_operation['operation_id'] in consumer:
            for pname in consumer[selected_operation['operation_id']]:
                if pname in producer:
                    for producer_operation_id in producer[pname]:
                        producer_operation = next(op for op in operations if op['operation_id'] == producer_operation_id)
                        producer_parameters = generate_parameter_values([producer_operation])[
                            producer_operation_id]

                        response, path, query_params, body_params = execute_operations(base_url, producer_operation, producer_parameters)
                        if (selected_operation['method'] in ["post", "get"]) and 200 <= response.status_code < 300:
                            try:
                                extract_response_values(response.json(), producer_operation)
                            except Exception:
                                pass

        response, path, query_params, body_params = execute_operations(base_url, selected_operation, selected_parameters)
        if (selected_operation['method'] in ["post", "get"]) and 200 <= response.status_code < 300:
            try:
                extract_response_values(response.json(), selected_operation)
            except Exception:
                pass
        update_q_table(q_table, alpha, gamma, selected_operation, selected_parameters, response, path, query_params, body_params)

        copied_operation = copy.deepcopy(selected_operation)
        copied_parameters = copy.deepcopy(selected_parameters)
        mutated_params, mutated_ops= perform_parameter_mutation(copied_parameters, copied_operation)
        execute_operations(base_url, mutated_ops, mutated_params)


        adapt_testing_strategy(iteration, max_iterations_without_improvement)
        iteration += 1

if __name__ == "__main__":
    base_url = sys.argv[2]
    EPSILON = [0.1]
    ss = [None]
    total_n = [0]
    key_matched = {}
    post_produced = {}
    previous_request = {}
    response_values = {}
    cached_media_type = {}
    q_table_param_values = {}
    http_500_details = {}
    producer = {}
    consumer = {}
    q_value = {}
    MUTATION_RATE = 0.1
    main()
    report_http_500_errors()
