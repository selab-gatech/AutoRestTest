import copy

from .reference_definition import ReferenceDefinition
from .variable_definition import VariableDefinition


class SequenceOrigin:
    ODG = 'ODG'
    RESOURCE_MERGING = 'RESOURCE_MERGING'
    DYNAMIC_FIX_ADD = 'DYNAMIC_FIX_ADD'
    DYNAMIC_FIX_DELETE = 'DYNAMIC_FIX_DELETE'


class Sequence:
    def __init__(self, requests=[]):
        self.requests = list(requests)
        self.def_vars = {}
        self.ref_vars = {}
        self.combination_targets = {}
        self.value_source = {}
        self.request_parameter_targets = {}
        self.origin = None
        self.sequence_trace = []
        self.method_names = set()
        self.concrete_value = []
        for req in self.requests:
            self.method_names.add(req.method_name)

    def sub_sequence(self, sub_sequence_index=[]):
        # copy whole sequence
        if len(sub_sequence_index) == len(self.requests):
            return self.duplicate()
        requests = []
        # if only one sub seq, we do not consider refs or defs
        if len(sub_sequence_index) == 1:
            for index in sub_sequence_index:
                requests.append(self.requests[index])
            return Sequence(requests)
        defs = {}
        refs = {}
        current_index = 0
        for index in sub_sequence_index:
            # we do not consider the first reference
            if index in self.ref_vars and current_index != 0:
                refs[current_index] = copy.copy(self.ref_vars[index])
            if index in self.def_vars:
                defs[current_index] = copy.copy(self.def_vars[index])
            current_index += 1
            requests.append(self.requests[index])
        seq = Sequence(requests)
        seq.def_vars = defs
        seq.ref_vars = refs
        return seq

    def add_method(self, method):
        self.requests.append(method)
        self.method_names.add(method.method_name)

    def add_def(self, index, parameter):
        defs = self.def_vars.get(index, None)
        if defs is None:
            defs = VariableDefinition()
        defs.add_definition(parameter)
        self.def_vars[index] = defs

    def add_ref(self, index, method, required_parameter, feed_parameter):
        refs = self.ref_vars.get(index, None)
        if refs is None:
            refs = ReferenceDefinition(method, {})
        refs.add_reference(required_parameter, feed_parameter)
        self.ref_vars[index] = refs

    def get_ref(self, index):
        refs = self.ref_vars.get(index, ReferenceDefinition(self.requests[index], {}))
        return refs

    def set_method(self, index, method):
        self.requests[index] = method

    def duplicate(self):
        seq = Sequence(self.requests)
        seq.ref_vars = copy.copy(self.ref_vars)
        seq.def_vars = copy.copy(self.def_vars)
        return seq

    def to_str(self):
        res = ''
        for method in self.requests:
            res += method.method_name + ' ->'
        res += '\n'
        for k, v in enumerate(self.def_vars):
            res += f'{v}:{self.def_vars[v]}\n'
        for k, v in enumerate(self.ref_vars):
            res += f'{v}:{self.ref_vars[v]}\n'
        return res

    def to_str_sequence(self):
        res = ''
        for method in self.requests:
            res += method.method_name + ' ->'
        res += '\n'
        return res

    def __eq__(self, other):
        if len(other.requests) != len(self.requests) or len(other.ref_vars.keys()) != len(self.ref_vars.keys()) or len(
                other.def_vars.keys()) != len(
            self.def_vars.keys()):
            return False
        for index in range(len(self.requests)):
            if self.requests[index] != other.requests[index]:
                return False
        for key in self.ref_vars.keys():
            if self.ref_vars[key] != other.ref_vars[key]:
                return False
        for key in self.def_vars.keys():
            if self.def_vars[key] != other.def_vars[key]:
                return False
        return True

    def __len__(self):
        return len(self.requests)

    def get_request_parameter_by_index(self, index):
        # return nominal request parameter set
        return self.requests[index].get_nominal_request_parameter()

    def has_method(self, method_name):
        return method_name in self.method_names

    def slice_by_method_name(self, method_name):
        ref_dict = {}
        def_dict = {}
        reqs = []
        for ind in range(len(self.requests)):
            method = self.requests[ind]
            if self.def_vars.__contains__(ind):
                def_dict[ind] = self.def_vars[ind]
            if self.ref_vars.__contains__(ind):
                ref_dict[ind] = self.ref_vars[ind]
            reqs.append(method)
            if method_name == method.method_name:
                break
        seq = Sequence(reqs)
        seq.def_vars = def_dict
        seq.ref_vars = ref_dict
        return seq

    def __hash__(self):
        return hash(self.to_str())

    def __str__(self):
        return self.to_str()

    def __getitem__(self, item):
        return self.requests[item]
