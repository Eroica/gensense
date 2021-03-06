"""This module implements a `Sentence' class that is used to store a
sentence and its vector space representation of each

"""

import gensim
import numpy
import scipy.spatial
import math

from copy import deepcopy
from collections import OrderedDict
from nltk.corpus import stopwords

DEBUG = True

# STOP_WORDS = set("for a an of the and to in".split())
STOP_WORDS = stopwords.words("english")

MODEL = gensim.models.word2vec.Word2Vec.load("share/vectors.bin")

class Sentence(OrderedDict):
    """`Sentence' is an object representing a sentence. It is an
    OrderedDict in which every word is a key and that word's vector
    space representation its value.

    """

    def __init__(self, sentence, model=MODEL, strict=False):
        """Creates a `Sentence' object by evaluating a string
        `sentence'. For each word in `sentence', that word's vector
        space representation is looked up in `model'. If it exists,
        they are appended to `self', unless `strict' is set to True,
        then this process will abort.

        :Example:

        >>> s = gensense.sentence.Sentence("this is an example")
        >>> s
        Sentence([('this', [1, 1, 1]), ('is', [1, 0, 1]), ('an', [0, 0, 1]), ('example', [0, 1, 0])])

        :param sentence: A sentence string
        :type sentence: str

        :param model:
            A model created by gensim.models.word2vec.Word2Vec()
        :type model : gensim.models.word2vec.Word2Vec

        :param strict=False:
            (Optional) specifies whether words not found in `model'
            should be ignored. If True, Sentence creation will abort.
        :type strict=False: bool

        :rtype: Sentence
        """

        super(Sentence, self).__init__()

        for word in sentence.split():
            try:
                self[word] = deepcopy(model[word])
            except KeyError:
                if not strict:
                    print ("Word `" + word +
                           "' was not found in model! Ignoring ...")
                    pass
                else:
                    print ("Word `" + word +
                           "' was not found in model! Aborting ...")
                    raise Exception()

        self.clusters = []
        self.cluster_sums = []

        self.clusterize()

    def __str__(self):
        """The __str__() method is overwritten to only return the
        sentence's words (thus ignoring each word's vector).

        :Example:

        >>> s = gensense.sentence.Sentence("this is an example")
        >>> print s
        this is an example

        """

        return " ".join((x for x in self.keys()))

    def __mul__(self, i):
        """
        """

        for vector in self.values():
            vector *= i

        return self

    def __rmul__(self, i):
        """
        """

        return self.__mul__(self, i)


    def removeStopWords(self, stop_words=STOP_WORDS):
        """Iterates over all words in `self' and removes those found in
        `stop_words'. By default, those stop words found in NLTK's
        corpus of English stop words are used.

        This method operates on this object's state and does not
        return anything.

        :param stop_words:
            List of strings of words that can be removed
        :type stop_words: list

        :Example:

        >>> s = gensense.sentence.Sentence("this is an example")
        >>> print s
        this is an example
        >>> s.removeStopWords()
        >>> print s
        example
        """

        for word in self.keys():
            if word in stop_words:
                self.pop(word)

                if DEBUG:
                    print "`" + word + "' has been removed from the sentence."

    def clusterize(self):
        """Maps each word of a sentence to an appropriate cluster by
        using a ``Chinese Restaurant Problem'' algorithm.
        """

        # The first word immediately creates a first cluster
        first_pair = tuple(self.items()[0])
        self.clusters = [[first_pair[0]]]
        self.cluster_sums = [deepcopy(first_pair[1])]

        # With a probability of 1.0 / (1 + len(self.clusters)),
        # a new cluster is created
        rands = numpy.random.rand(len(self))
        pnew = 1.0

        for i, word in enumerate(self.items()[1:]):
            maxSim = float("inf")

            # Calculates the similarity of `word' with each cluster, and saves
            # the cluster with the highest similarity
            for j, c in enumerate(self.cluster_sums):
                sim = abs(scipy.spatial.distance.cosine(word[1], c))
                sim = float("inf") if math.isnan(sim) else sim

                if sim <= maxSim:
                    maxSim = sim
                    maxClusterIndex = int(j)

            if maxSim > pnew and rands[i+1] < pnew:
                # Create a new cluster, and update the cluster vector
                self.clusters.append([deepcopy(word[0])])
                self.cluster_sums.append(deepcopy(word[1]))
                pnew = 1.0 / (1 + len(self.clusters))
            else:
                # Do not create a new cluster, but add `word' to an
                # existing one, and update the cluster vector
                self.clusters[maxClusterIndex].append(word[0])
                self.cluster_sums[maxClusterIndex] += word[1]
