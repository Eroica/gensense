import gensim

DEBUG = True
STOP_LIST = set("for a an of the and to in".split())


class Sentence(list):
    """`Sentence' is an object representing a sentence. It consists of a
    list of words. Every word is represented by the word as a string,
    and its vector space representation in a given model.

    """

    def __init__(self, sentence, model, strict=False):
        """Creates a `Sentence' object. A sentence is represented in a
        list of string-value pairs: `string' is the individual word,
        and `value' its vector space representation in a given model.

        For instance:

        [   ("this",    [0, 0, 1]),
            ("is",      [0, 1, 0]),
            ("an",      [1, 0, 0]),
            ("example", [1, 1, 1])  ]

        @params:
            sentence : str
                A string

            model : gensim.models.word2vec.Word2Vec
                A model created by gensim.models.word2vec.Word2Vec()

            strict=False : boolean
                (Optional) specifies whether words not found in model
                should be ignored. If True, Sentence creation will
                abort.

        @returns:
            Sentence : Sentence
        """

        for word in sentence.split():
            try:
                self.append((word, model[word]))
            except KeyError:
                if not strict:
                    print ("Word `" + word +
                           "' was not found in model! Ignoring ...")
                    pass
                else:
                    print ("Word `" + word +
                           "' was not found in model! Aborting ...")
                    return None

    def __str__(self):
        """Overrides the __str__() method to only return the sentence as
        a list of words (thus ignoring the vector space representation).
        """

        return str([x[0] for x in self])

    def removeStopWords(self, stop_list=STOP_LIST):
        """Iterates over all words in `self' and removes those found in
        `stop_list'. This method operates on the object's state and does
        not return anything.

        @params:
            stop_list : list
                List of strings of words
        @returns:
            None
        """
        for i, v in enumerate(self):
            if v[0] in stop_list:
                print ("`" + str(self.pop(i)[0]) +
                       "' has been removed from the sentence.")
