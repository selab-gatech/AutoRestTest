from model.method import Method
from pydblite import Base
from fuzzer.util import resolve_json_value


class APIDB:
    def __init__(self, method: Method):
        self.method_name = method.method_name
        self.method_db = None
        self.method = method
        self.sub_db = []

    def is_simple_entity(self):
        return self.method_db and len(self.sub_db) == 0

    def create_simple_entity_db(self):
        self.method_db = Base('{}.pdl'.format(self.method_name))
        if self.method_db.exists():
            self.method_db.open()

    def remove_variable_replacement_name(self, name):
        name = name.replace("%{APIGen#", "")
        name = name.replace("}", "")
        return name

    def cluster_by_path(self, data):
        result = {}
        for concrete_key in data:
            key = concrete_key
            if "." in concrete_key:
                key = concrete_key[:concrete_key.rindex(".")]
            pair = (concrete_key, data[concrete_key])
            if not key in result:
                result[key] = []
            result[key].append(pair)
        return result

    def cluster_response_by_path(self, data):
        return self.cluster_by_path(data)

    def insert_data(self, concrete_request_parameter_list, request_mapping_dict, response_data):
        request_value_dict = {}
        replace_result = {}
        # extract replace mapping dict
        for component in request_mapping_dict:
            for result in request_mapping_dict[component]:
                if result["replaceStatus"] == 'success':
                    replace_result[result['variableName']] = result['replaceValue']
        # parse concrete request value
        for request_parameter_pair in concrete_request_parameter_list:
            request_parameter, concrete_request_value = request_parameter_pair
            single_concrete_request_value_dict = dict()
            resolve_json_value(request_parameter.name, concrete_request_value, single_concrete_request_value_dict)
            filtered_parameter = {}
            for key in single_concrete_request_value_dict:
                val = single_concrete_request_value_dict[key]
                if "id" in str.lower(key) and isinstance(val, str):
                    # replace with concrete request value
                    if self.remove_variable_replacement_name(val) in replace_result:
                        filtered_parameter[key] = replace_result[
                            self.remove_variable_replacement_name(val)]
                    else:
                        filtered_parameter[key] = val
            if len(filtered_parameter) > 0:
                request_value_dict[request_parameter] = filtered_parameter
        # filter response
        filtered_response_dict = {}
        for key in response_data:
            value = response_data[key]
            if isinstance(value, dict) or isinstance(value, list) or value is None or isinstance(value, bool):
                continue
            if "id" in str.lower(key):
                filtered_response_dict[key] = value
        # check empty response
        if len(filtered_response_dict) == 0:
            return
        clustered_response = self.cluster_response_by_path(filtered_response_dict)
        a = 2


class RelationReasoner:
    def __init__(self):
        self.dbs = {}

    def create_db(self, method: Method):
        self.dbs[method] = APIDB(method)
        self.dbs[method].create_simple_entity_db()

    def update_or_create_db_by_data(self, concrete_value, variable_replace_map, method: Method, data):
        if not (method in self.dbs):
            self.create_db(method)
        self.dbs[method].insert_data(concrete_value, variable_replace_map, data)
