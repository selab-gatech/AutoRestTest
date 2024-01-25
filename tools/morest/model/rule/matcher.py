from model.method import Method
from model.rule.blacklist import BLACK_LIST
from model.rule.concrete_rule import ConcreteRule
from model.rule.substr_rule import SubStringRule
from model.rule.edit_distance_rule import EditDistanceRule
from model.util.path_resource_util import WordUtil
from model.rule.resource_path_rule import ResourcePathRule


class RuleMatcher:
    @staticmethod
    def match(from_method=Method, to_method=Method):
        if ConcreteRule.match(from_method,
                              to_method):
            to_method.feed_from_method.add(from_method)
            to_method.required_feed_parameter[from_method.method_signature] = ConcreteRule.get_params(
                from_method,
                to_method)
            from_method.output_to_method.add(to_method)
            return True
        if SubStringRule.match(from_method, to_method):
            to_method.feed_from_method.add(from_method)
            to_method.required_feed_parameter[from_method.method_signature] = SubStringRule.get_params(
                from_method,
                to_method)
            from_method.output_to_method.add(to_method)
            return True
        # if ResourcePathRule.match(from_method, to_method):
        #     to_method.feed_from_method.add(from_method)
        #     from_method.output_to_method.add(to_method)
        #     to_method.required_feed_parameter[from_method.method_signature] = {}
        #     return True
        # if StemRule.match(from_method.response_parameter_name, to_method.request_parameter_name):
        #     to_method.feed_from_method.add(from_method)
        #     to_method.required_feed_parameter[from_method.method_signature] = StemRule.get_params(
        #         from_method.response_parameter_name,
        #         to_method.request_parameter_name)
        #     from_method.output_to_method.add(to_method)
        #     return True
        if EditDistanceRule.match(from_method, to_method):
            to_method.feed_from_method.add(from_method)
            to_method.required_feed_parameter[from_method.method_signature] = EditDistanceRule.get_params(
                from_method,
                to_method)
            from_method.output_to_method.add(to_method)
            return True
        return False
