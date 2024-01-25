from nltk.stem import PorterStemmer
from re import finditer


def camel_case_split(identifier):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]


class StemRule:
    @staticmethod
    def match(parameters_set1=set(), parameters_set2=set()):
        stemmer = PorterStemmer()
        for p1 in parameters_set1:
            for p2 in parameters_set2:
                if tuple(stemmer.stem(p1.lower())) == tuple(stemmer.stem(p2.lower())):
                    return True
        return False

    @staticmethod
    def get_params(parameters_set1=set(), parameters_set2=set()):
        stemmer = PorterStemmer()
        parameter_map = {}
        res = set()
        for p1 in parameters_set1:
            for p2 in parameters_set2:
                if tuple(stemmer.stem(p1.lower())) == tuple(stemmer.stem(p2.lower())):
                    parameter_map[p1] = p2
        return parameter_map
