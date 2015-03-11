"""

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
        """Creates a `Sentence' object. A sentence is represented as an
        OrderedDict in which every word is a key with the corresponding
        vector space representation of that word as a value.

        :Example:

        >>> s = gensense.sentence.Sentence("this is an example")
        >>> s
        Sentence([('this', [1, 1, 1]), ('is', [1, 0, 1]), ('an', [0, 0, 1]), ('example', [0, 1, 0])])

        :params: sentence: A sentence string
        :type: sentence: str

        :params: model:
            A model created by gensim.models.word2vec.Word2Vec()
        :type: model : gensim.models.word2vec.Word2Vec

        :params: strict=False:
            (Optional) specifies whether words not found in `model'
            should be ignored. If True, Sentence creation will abort.
        :type: strict=False: bool

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
                    return None

        self.clusters = []
        self.cluster_sums = []

        self.clusterize()

    def __str__(self):
        """The __str__() method is overwritten to only return the
        sentence's words (thus ignoring each word's vector).
        """

        return " ".join((x for x in self.keys()))

    def removeStopWords(self, stop_words=STOP_WORDS):
        """Iterates over all words in `self' and removes those found in
        `stop_words'. By default, those stop words found in NLTK's
        corpus of English stop words are used.

        This method operates on this object's state and does not
        return anything.

        :params: stop_words:
            List of strings of words that can be removed
        :type: stop_words: list
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

        # With a probability of 1/(n + len(clusters)), a new cluster is created
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


def sv_additive(sentence):
    """Returns a sentence vector where sv_i is the sum of each i-th
    component of every word vector.

    :rtype: float
    """

    return reduce(lambda x, y: x + y, sentence.values())


def sv_multiplicative(sentence):
    """Returns a sentence vector where sv_i is the product of all i-th
    components of every word vector.

    :rtype: float
    """

    return reduce(lambda x, y: numpy.multiply(x, y),
                  sentence.values())

def cosine_to_integer(grade):
    """Since Michell's and Lapata's original evaluation used numbers
    from 1 to 7 to grade sentence similarity, this method takes a
    cosine similarity (a number between 0.0 and 1.0) and maps them
    to either 1, 2, 3, 4, 5, 6, or 7.

    :rtype: int

    :Example:

    >>> gensense.sentence.cosine_to_integer(0.45)
    >>> 4.0
    """

    # grads = map(lambda x: x * (1.0/7), [x for x in range(0, 8)])
    return int(math.ceil(grade * 7))


def similarity(sentence1, sentence2, sv_function=sv_additive):
    """Returns the cosine similarity between two sentences, calculated
    by comparing their sentence vectors with each other.

    This function requires a function that explains how to calculate
    each sentence's sentence vector. By default, an additive sentence
    vector is used, but any other function that returns a sentence
    vector (e.g., by multiplying each component instead of adding them)
    can be used as a 3rd argument.

    :params: sentence1: A sentence
    :type: sentence1: Sentence

    :params: sentence2: Another sentence
    :type: sentence2: Sentence

    :params: sv_function: Function calculating sentence vector
    :type: sv_function: function

    :returns: Number between 0.0 and 1.0 (cosine similarity)
    :rtype: float
    """

    return 1 - scipy.spatial.distance.cosine(sv_function(sentence1),
                                             sv_function(sentence2))