import copy


class ReferenceDefinition:
    def __init__(self, method, refs={}):
        self.refs = refs
        self.method = method

    def add_reference(self, required_parameter, feed_parameter):
        ref = self.refs.get(feed_parameter, set())
        ref.add(required_parameter)
        self.refs[feed_parameter] = ref

    def get_refs(self):
        return self.refs

    def __contains__(self, item):
        return self.refs.__contains__(item)

    def __eq__(self, other):
        return str(self) == str(other)

    def __copy__(self):
        return ReferenceDefinition(self.method, copy.deepcopy(self.refs))

    def __str__(self):
        return "Ref: " + str(self.refs) + str(self.method)

    def __getitem__(self, item):
        return self.refs[item]
