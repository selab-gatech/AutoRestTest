class VariableDefinition:
    def __init__(self, defs=set()):
        self.defs = set(defs)

    def add_definition(self, definition):
        self.defs.add(definition)

    def __contains__(self, item):
        return item in self.defs

    def __eq__(self, other):
        return self.defs == other.defs

    def __copy__(self):
        return VariableDefinition(set(self.defs))

    def __len__(self):
        return len(self.defs)

    def __iter__(self):
        return iter(self.defs)

    def __str__(self):
        return "Def: " + str(self.defs)
