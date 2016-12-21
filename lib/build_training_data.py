

class BuildData:
    """ Build training data via search text in corpus and label. """


    def __init__(self, rawTermsPath):
        f = open(rawTermsPath, "r", encoding="utf-8")
        self._terms = f.read()
