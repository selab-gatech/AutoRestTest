def get_schema_type(schema: dict):
    if schema.__contains__('type'):
        return schema['type']
    if schema.__contains__('properties'):
        return 'object'
    if schema.__contains__('allOf') and isinstance(schema['allOf'], list):
        return 'allOf'
    if schema.__contains__('schema'):
        return 'schema'
    raise Exception(f'unknown schema type: {schema}')


def string_handler(instance, schema):
    if not isinstance(str(instance), str):
        raise Exception(f'{instance} is not string')


def number_handler(instance, schema):
    if not isinstance(instance, int) and not isinstance(instance, float):
        raise Exception(f'{instance} is not number')


def integer_handler(instance, schema):
    if int(instance) != instance:
        raise Exception(f'{instance} is not int')


def object_handler(instance, schema):
    if not isinstance(instance, dict):
        raise Exception(f'{instance} is not dict')
    # check required properties
    required_properties = []
    if schema.__contains__('required'):
        required_properties = schema['required']
    for prop in required_properties:
        if not (prop in instance):
            raise Exception(f'{prop} is required in {schema} while is not in {instance}')

    # check single properties
    properties = schema['properties']
    for prop in instance:
        if not (prop in properties):
            raise Exception(f'instance {instance}, property {prop} should be included in {properties}')
        validate(instance[prop], properties[prop])


def boolean_handler(instance, schema):
    if not (instance in ['false', 'true', True, False]):
        raise Exception(f'{instance} is not boolean')


def array_handler(instance, schema):
    # check array type
    if not isinstance(instance, list):
        raise Exception(f'{instance} is not array')
    array_items = schema['items']
    for elem in instance:
        validate(elem, array_items)


def all_of_handler(instance, schema):
    if not isinstance(instance, dict):
        raise Exception(f'{instance} is not allOf dict')
    all_of = schema['allOf']
    for elem in all_of:
        validate(instance, elem)


def schema_handler(instance, schema):
    if not schema.__contains__('schema'):
        raise Exception(f'{schema} does not contain schema property')
    validate(instance, schema['schema'])


def validate(instance, schema):
    data_type_handlers = {
        "string": string_handler,
        "number": number_handler,
        "integer": integer_handler,
        "object": object_handler,
        "array": array_handler,
        "boolean": boolean_handler,
        "allOf": all_of_handler,
        "schema": schema_handler
    }
    schema_type = get_schema_type(schema)
    data_type_handlers[schema_type](instance, schema)
