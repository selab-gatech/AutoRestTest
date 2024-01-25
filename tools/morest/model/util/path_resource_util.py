from nltk.stem import PorterStemmer
import difflib


class WordUtil:
    @staticmethod
    def extract_specific_resource_name(word):
        res = set()
        for target in word.split('-'):
            if len(target) > 0:
                res.add(target)
        for target in word.split('.'):
            if len(target) > 0:
                res.add(target)
        for target in word.split('_'):
            if len(target) > 0:
                res.add(target)
        return res

    @staticmethod
    def get_path_resource_list(path):
        stemmer = PorterStemmer()
        res = []
        target = path
        target = target.replace('{', '')
        target = target.replace('}', '')
        target = target.split('/')
        for word in target:
            if len(word) == 0:
                continue
            for split_word in WordUtil.extract_specific_resource_name(word):
                stemmed = stemmer.stem(split_word)
                res.append(stemmed)
        return res

    @staticmethod
    def match_path(first_path, second_path):
        threshold = 0.5
        matcher = difflib.SequenceMatcher()
        matcher.set_seq1(first_path)
        matcher.set_seq2(second_path)
        return threshold < matcher.ratio()
