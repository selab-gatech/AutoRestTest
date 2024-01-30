from nltk.stem import PorterStemmer


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
