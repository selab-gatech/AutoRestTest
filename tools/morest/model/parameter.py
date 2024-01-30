import re
import uuid
from nltk.stem.snowball import SnowballStemmer


class TargetStatus:
    EMPTY = 'EMPTY'
    NON_EMPTY = 'NON_EMPTY'
    TRUE = "TRUE"
    FALSE = "FALSE"
    MIN = "MIN"
    MAX = "MAX"
    MIDDLE = "MIDDLE"
    EXAMPLE = 'EXAMPLE'


class TargetType:
    ENUM = 'ENUM'
    NUM = 'NUM'
    ARRAY = "ARRAY"
    BOOL = "BOOL"
    STRING = "STRING"
    EXAMPLE = 'EXAMPLE'


class Parameter:
    def __init__(self, name, body={}, method_path=""):
        self.name = name
        self.raw_body = body
        self.has_schema = False
        self.schema = None
        self.method_path = method_path
        self.attribute_path_dict = {}
        self.parameter_id = "h" + uuid.uuid4().__str__()
        self.parameter_names = set()
        self.parameter_names.add(name)
        self.parameter_body_tuple = {}
        self.parameter_body_tuple[name] = [(name, body)]
        # record_all_attribute
        self.attributes = set([name])
        # target consists of (type, path, value)
        self.cover_targets = set()
        # if name == 'delete_lun_groups_request':
        #     print()
        self.attribute_path_dict[name] = set([name])
        self.parse(body)
        # extend parameter name by path resource
        if self.attribute_should_extend(name):
            self.parameter_names.remove(name)
            self.parameter_body_tuple[name].remove((name, body))
            canonical_name = self.get_canonical_name(name)
            tokens = self.tokenize_method_path(method_path)
            for token in tokens:
                attribute_path = token + '_' + canonical_name
                self.attribute_path_dict[attribute_path] = set([name])
                self.parameter_names.add(attribute_path)
                self.parameter_body_tuple[attribute_path] = [(attribute_path, body)]

    def parse(self, body={}):
        if body.__contains__("schema"):
            self.has_schema = True
            self.parse_schema(body["schema"])
        else:
            if body.__contains__("name"):
                name = body["name"]
                self.parameter_names.add(name)
                body_tuple_list = self.parameter_body_tuple.get(name, [])
                body_tuple_list.append((name, body))
                self.parameter_body_tuple[name] = body_tuple_list
                self.parse_schema(body)
            elif body.__contains__("properties"):
                self.has_schema = True
                self.schema = body
                self.parse_schema(self.schema["properties"])
            else:
                print("Body without name")

    def parse_schema(self, schema={}):
        if isinstance(schema, dict):
            if schema.__contains__("properties") or schema.__contains__("allOf") or schema.__contains__('name'):
                queue = [("", schema)]
                # handle allOf case
                if schema.__contains__("allOf"):
                    for item in schema["allOf"]:
                        queue.append(("", item))
                while len(queue) > 0:
                    elem = queue.pop()
                    prefix = elem[0]

                    elem = elem[1]
                    if elem.__contains__("properties"):
                        properties = elem["properties"]
                        # required_properties = []
                        # if elem.__contains__('required'):
                        #     required_properties = elem['required']
                        for name in properties.keys():
                            # if name in required_properties:
                            #     properties[name]['required'] = True
                            # check array type
                            is_array = False
                            if properties[name].__contains__("type") and properties[name]["type"] == "array":
                                is_array = True
                            # check prefix
                            if len(prefix) == 0:
                                child_prefix = prefix
                            else:
                                child_prefix = prefix
                                child_prefix += "."
                            self.parameter_names.add(name)
                            # add into attribute set
                            attribute_path = child_prefix + name
                            attribute_set = self.attribute_path_dict.get(name, set())
                            attribute_set.add(attribute_path)
                            body_tuple_list = self.parameter_body_tuple.get(name, [])
                            body_tuple_list.append((attribute_path, properties[name]))
                            self.parameter_body_tuple[name] = body_tuple_list
                            self.attributes.add(self.name + "." + attribute_path)
                            self.attribute_path_dict[name] = attribute_set
                            # if attribute_path == 'id':
                            #     print()
                            # extend property name by attribute path
                            if self.attribute_should_extend(name):
                                self.parameter_names.remove(name)
                                self.parameter_body_tuple[name].remove((attribute_path, properties[name]))
                                is_first_level_attribute = False
                                if len(prefix) != 0:
                                    prefix_tokens = [self.tokenize_attribute_path(child_prefix)]
                                else:
                                    prefix_tokens = self.tokenize_method_path(self.method_path)
                                    is_first_level_attribute = True

                                for prefix_token in prefix_tokens:
                                    if is_first_level_attribute:
                                        path = prefix_token + '_' + name
                                    else:
                                        path = prefix_token + '_' + name
                                    attribute_path_set = self.attribute_path_dict.get(path, set())
                                    attribute_path_set.add(attribute_path)
                                    self.attribute_path_dict[path] = attribute_path_set
                                    self.parameter_names.add(path)
                                    body_tuple_list = self.parameter_body_tuple.get(path, [])
                                    body_tuple_list.append((path, properties[name]))
                                    self.parameter_body_tuple[path] = body_tuple_list

                            # traverse inner properties
                            if isinstance(properties[name], dict):
                                if is_array:
                                    queue.append((child_prefix + name + "[0]", properties[name]))
                                else:
                                    queue.append((child_prefix + name, properties[name]))

                    if elem.__contains__("allOf"):
                        for e in elem["allOf"]:
                            queue.append((prefix, e))

                    if elem.__contains__("items"):
                        if not prefix.endswith('[0]'):
                            prefix += '[0]'
                        self.parameter_body_tuple[prefix] = [
                            (prefix, elem["items"])]
                        self.attribute_path_dict[prefix] = [prefix]
                        self.parameter_names.add(prefix)
                        queue.append((prefix, elem["items"]))
                    targets = self.cover_targets
                    # try:
                    #     int(self.name + 1)
                    #     full_path = prefix
                    # except Exception as ex:
                    #     if len(prefix) == 0:
                    #         full_path = self.name
                    #     else:
                    #         full_path = self.name + "." + prefix
                    #         self.attributes.add(full_path)
                    if len(prefix) == 0:
                        continue
                    full_path = prefix
                    # handle enum targets
                    if elem.__contains__('enum'):
                        for enum in elem["enum"]:
                            tar = (full_path, TargetType.ENUM, enum)
                            targets.add(tar)
                    elif elem.__contains__("type"):
                        element_type = elem['type']
                        # handle array type
                        if element_type == 'array':
                            targets.add((full_path.replace('[0]', ''), TargetType.ARRAY, TargetStatus.EMPTY))
                            targets.add((full_path.replace('[0]', ''), TargetType.ARRAY, TargetStatus.NON_EMPTY))
                        # handle string type
                        elif element_type == 'string':
                            if elem.__contains__('example'):
                                tar = (full_path, TargetType.EXAMPLE, TargetStatus.EXAMPLE)
                                targets.add(tar)
                            # if elem.__contains__("minLength"):
                            #     targets.add((full_path, TargetType.STRING, TargetStatus.MIN))
                            # if elem.__contains__("maxLength"):
                            #     targets.add((full_path, TargetType.STRING, TargetStatus.MAX))
                            # if elem.__contains__("minLength") and elem.__contains__("maxLength"):
                            #     targets.add((full_path, TargetType.STRING, TargetStatus.MIDDLE))
                        elif element_type == 'boolean':
                            targets.add((full_path, TargetType.BOOL, TargetStatus.TRUE))
                            targets.add((full_path, TargetType.BOOL, TargetStatus.FALSE))
                        elif element_type == 'integer':
                            if elem.__contains__('example'):
                                tar = (full_path, TargetType.EXAMPLE, TargetStatus.EXAMPLE)
                                targets.add(tar)
                            # if elem.__contains__("minimum"):
                            #     targets.add((full_path, TargetType.NUM, TargetStatus.MIN))
                            # if elem.__contains__("maximum"):
                            #     targets.add((full_path, TargetType.NUM, TargetStatus.MAX))
                            # if elem.__contains__("minimum") and elem.__contains__("maximum"):
                            #     targets.add((full_path, TargetType.NUM, TargetStatus.MIDDLE))
                        elif element_type == 'number':
                            if elem.__contains__('example'):
                                tar = (full_path, TargetType.EXAMPLE, TargetStatus.EXAMPLE)
                                targets.add(tar)
                            # if elem.__contains__("minimum"):
                            #     targets.add((full_path, TargetType.NUM, TargetStatus.MIN))
                            # if elem.__contains__("maximum"):
                            #     targets.add((full_path, TargetType.NUM, TargetStatus.MAX))
                            # if elem.__contains__("minimum") and elem.__contains__("maximum"):
                            #     targets.add((full_path, TargetType.NUM, TargetStatus.MIDDLE))
                        # elif element_type == 'object':
                        #     pass
                        # else:
                        #     print()

                    # for property in elem.keys():
                    #     item = elem[property]
                    #     if isinstance(item, dict):
                    #         queue.append((prefix, item))
        else:
            return

    def get_canonical_name(self, name):
        stemmer = SnowballStemmer("english")
        stemmed_word = stemmer.stem(name)
        return stemmed_word

    def tokenize_method_path(self, method_path: str):
        tokens = method_path.split('/')
        result = []
        version_result = []
        has_version = False
        pattern = re.compile('^v[0-9]+$')
        for token in tokens:
            # filter in path parameter
            if '{' in token:
                continue

            # remove empty string
            if len(token) == 0:
                continue
            # check whether has version
            if pattern.match(token) is not None:
                has_version = True
                continue
            token = self.get_canonical_name(token)
            # ignore base path for service
            if has_version:
                version_result.append(token)
            else:
                result.append(token)
        if has_version and len(version_result) > 0:
            return version_result
        return result

    def tokenize_attribute_path(self, attribute_path: str):

        if len(attribute_path) == 0:
            return ''
        tokens = attribute_path.replace('[0]', '').split('.')
        result = []
        for token in tokens:
            if len(token) == 0:
                continue
            result.append(self.get_canonical_name(token))
        return result[-1]

    def attribute_should_extend(self, attribute_name):
        targets = ['id', 'name']
        name = self.get_canonical_name(attribute_name)
        return name in targets

    def __str__(self):
        return f'({self.name}, {self.schema})'
