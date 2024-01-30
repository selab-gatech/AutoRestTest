import json
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def json_compare(obj1, obj2):
    '''
    :param obj1: a string object that is ideal to be in json format
    :param obj2: a string object that is ideal to be in json format
    :return similarity: a measure of similarity between two objects ranging from 0 to 1
    '''
    obj1_is_json = True
    obj2_is_json = True
    try:
        obj1_json = json.loads(obj1)
    except:
        obj1_is_json = False

    try:
        obj2_json = json.loads(obj2)
    except:
        obj2_is_json = False

    if not obj1_is_json and not obj2_is_json:
        return similar(obj1, obj2)     # a typical 0.6 should be considered as match

    elif obj2_is_json and obj1_is_json:
        pass   # proper measure of similarities in two json file
        # for now, we temporarily assume that the two files are in the same format.
        return 1

    else:
        # one is json, the other is not. This is clear that the two objects are not the same.
        return 0
