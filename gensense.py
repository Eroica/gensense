import gensim
import numpy
import scipy.spatial
import math
from copy import deepcopy
from collections import OrderedDict
import itertools

DEBUG = True
STOP_LIST = set("for a an of the and to in".split())


class Sentence(OrderedDict):
    """`Sentence' is an object representing a sentence. It is an
    OrderedDict in which every word is a key and its vector space
    representation its value.

    """

    def __init__(self, sentence, model, strict=False):
        """Creates a `Sentence' object. A sentence is represented as an
        OrderedDict in which every word is a key with the corresponding
        vector space representation of that word as a value.

        For instance:

        [   "this":    [0, 0, 1]),
            "is":      [0, 1, 0]),
            "an":      [1, 0, 0]),
            "example": [1, 1, 1])  ]

        @params:
            sentence : str
                A string

            model : gensim.models.word2vec.Word2Vec
                A model created by gensim.models.word2vec.Word2Vec()

            strict=False : boolean
                (Optional) specifies whether words not found in `model'
                should be ignored. If True, Sentence creation will
                abort.

        @returns:
            Sentence : Sentence
        """

        super(Sentence, self).__init__()

        for word in sentence.split():
            try:
                self[word] = deepcopy(model[word])
                # tmp.append((word, model[word]))
            except KeyError:
                if not strict:
                    print ("Word `" + word +
                           "' was not found in model! Ignoring ...")
                    pass
                else:
                    print ("Word `" + word +
                           "' was not found in model! Aborting ...")
                    return None

        self.sentenceVector = reduce(lambda x, y: numpy.multiply(x,y),
                                     [self[x] for x in self])
        self.clusters = []
        self.cluster_sums = []

        self.clusterize()

    def __str__(self):
        """Overrides the __str__() method to only return the sentence as
        a list of words (thus ignoring the vector space representation).

            Sentence([("this", [0, 0, 1]), ("is", [0, 1, 0]),
             ("an", [1, 0, 0]), ("example", [1, 1, 1])])

            => "this is an example"
        """

        return " ".join((x for x in self.keys()))

    def removeStopWords(self, stop_list=STOP_LIST):
        """Iterates over all words in `self' and removes those found in
        `stop_list'. This method operates on this object's state and
        does not return anything.

        @params:
            stop_list : list
                List of strings of words
        @returns:
            None
        """
        for word in self.keys():
            if word in stop_list:
                self.pop(word)
                print "`" + word + "' has been removed from the sentence."

        self.updateSentenceVector()

    def updateSentenceVector(self):
        """
        """

        self.sentenceVector = reduce(lambda x, y: numpy.multiply(x,y),
                                     [self[x] for x in self])

    def sv_additive(self):

        sv = reduce(lambda x, y: x + y, self.values())

        return sv


    def sv_multiplicative(self):

        sv = reduce(lambda x, y: numpy.multiply(x, y),
                    self.values())

        return sv


    def sv_kintsch(self):
        pass

    def clusterize(self):
        """
        """

        first_pair = list(self.items()[0])

        self.clusters = [[first_pair[0]]]
        self.cluster_sums = [deepcopy(first_pair[1])]

        rands = numpy.random.rand(len(self))
        pnew = 1.0

        for i, word in enumerate(self.items()[1:]):
            maxSim = float("inf")

            for j, c in enumerate(self.cluster_sums):
                sim = abs(scipy.spatial.distance.cosine(word[1], c))
                sim = float("inf") if math.isnan(sim) else sim

                if sim <= maxSim:
                    maxSim = sim
                    maxClusterIndex = int(j)

            if maxSim > pnew and rands[i+1] < pnew:
                self.clusters.append([deepcopy(word[0])])
                self.cluster_sums.append(deepcopy(word[1]))
                pnew = 1.0 / (1 + len(self.clusters))
            else:
                self.clusters[maxClusterIndex].append(word[0])
                self.cluster_sums[maxClusterIndex] += word[1]


    def returnVectors(self):
        return [w[1] for w in self]

model = gensim.models.word2vec.Word2Vec.load("vectors.bin")

def similarity(sentence1, sentence2, ):
    return 1 - scipy.spatial.distance.cosine(sentence1, sentence2)