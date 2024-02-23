import numpy as np


def extract_dict_value(path="", data={}, value_collection={}):
    prefix = ""
    if len(path) > 0:
        prefix = path + "."
    for key in data.keys():
        item = data[key]
        path = prefix + key
        resolve_json_value(path, item, value_collection)


def resolve_json_value(path, item, value_collection={}):
    if isinstance(item, str):
        value_collection[path] = item
    elif isinstance(item, int):
        value_collection[path] = item
    elif isinstance(item, float):
        value_collection[path] = item
    elif item == None:
        if len(path) > 0:
            value_collection[path] = None
    elif isinstance(item, dict):
        if len(path) > 0:
            value_collection[path] = item
        extract_dict_value(path, item, value_collection)
    elif isinstance(item, list):
        if len(path) > 0:
            value_collection[path] = item
        # FIXME: ignore huge list item
        np.random.shuffle(item)
        for i, val in enumerate(item):
            resolve_json_value(path + '[' + str(i) + ']', val, value_collection)
    else:
        raise Exception(f'Unknown parameter {path} {str(item)}')


def fetch_object_value_by_attribute_path(attribute_path: str, obj: dict):
    tokens = attribute_path.split('.')
    content = obj
    for token in tokens:
        content = content[token]
    return content
