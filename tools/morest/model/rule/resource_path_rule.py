from model.util.path_resource_util import WordUtil


class ResourcePathRule:
    @staticmethod
    def match(from_method, to_method):
        from_method_resource_path = WordUtil.get_path_resource_list(from_method.method_path)
        to_method_resource_path = WordUtil.get_path_resource_list(to_method.method_path)
        # identical path should be considered in CRUD
        if from_method_resource_path == to_method_resource_path:
            return False
        # should have valid CRUD relation
        if from_method.crud >= to_method.crud:
            return False

        # the following code guarantee the prefix of resource path
        if len(from_method_resource_path) > len(to_method_resource_path):
            return False
        if from_method_resource_path in [to_method_resource_path[i:len(from_method_resource_path) + i] for i in
                                         range(len(to_method_resource_path))]:
            print('resource path infer', from_method, '->', to_method)
            return True
        return False
