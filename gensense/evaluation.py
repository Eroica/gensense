from __future__ import print_function
from copy import deepcopy
from collections import defaultdict
from tabulate import tabulate

from corpus import EvaluationCorpus
from sentence import MODEL
from sentence import Sentence

import scipy.spatial
import numpy
import math
import nltk

ML_CORPUS = EvaluationCorpus()

# These weights are used in sv_weightadd() to influence the word vector
# depending on its part-of-speech tag. 'NN', 'PRP', etc. are the tags
# used by NLTK's `pos_tag()'. If a word does not belong to a tag found
# in `WEIGHTS', a default weight of 0.1 is used (can be changed in the
# 2nd line).
WEIGHTS = {'NN': 0.5, 'PRP': 0.5, 'VBP': 0.4, 'VBN': 0.4}
WEIGHTS = defaultdict(lambda: 0.1, WEIGHTS)

def cosine_to_integer(grade):
    """Since Michell's and Lapata's original evaluation used numbers
    from 1 to 7 to grade sentence similarity, this method takes a
    cosine similarity (a number between 0.0 and 1.0) and maps them
    to either 1, 2, 3, 4, 5, 6, or 7.

    :rtype: int

    :Example:

    >>> gensense.evaluate.cosine_to_integer(0.45)
    >>> 4.0
    """

    # grads = map(lambda x: x * (1.0/7), [x for x in range(0, 8)])
    assert grade > -1.0 and grade < 1.0

    return int(math.ceil(abs(grade * 7)))



def sv_add(sentence):
    """Returns a sentence vector where sv_i is the sum of each i-th
    component of every word vector.

    :params: sentence: The sentence whose sentence vector to calculate
    :type: sentence: Sentence
    :rtype: float
    """

    return reduce(lambda x, y: x + y, sentence.values())


def sv_weightadd(sentence, weights=WEIGHTS):
    """Returns a weightened sentence vector where each individual word
    vector is multiplied by a factor depending on their POS tag. Weights
    are determined from a defaultdict of NLTK POS tags, e.g.

    >>> from collections import defaultdict
    >>> weights = {'NN': 0.9, 'VNB': 0.5}
    >>> weights = defaultdict(lambda: 0.1, weights)

    A defaultdict is used so that all words are multiplied by a standard
    factor (in this case: 0.1), and you only have to specify those tags
    that need another factor.

    All weightened word vectors are then sumed up into a sentence
    vector.

    :params: sentence: The sentence whose sentence vector to calculate
    :type: sentence: Sentence

    :params: weights: Factors describing how to weight a word
    :type: weights: defaultddict

    :rtype: float
    """

    tagged_words = nltk.pos_tag(str(sentence).split())
    word_vectors = [deepcopy(vector) for vector in sentence.values()]

    # In the following loop, word[0] is the word and word[1] is the tag
    # as determined by NLTK's `pos_tag()'.
    for i, word in enumerate(tagged_words):
        word_vectors[i] *= WEIGHTS[word[1]]

    return reduce(lambda x, y: x + y, word_vectors)


def sv_multiply(sentence):
    """Returns a sentence vector where sv_i is the product of all i-th
    components of every word vector.

    :params: sentence: The sentence whose sentence vector to calculate
    :type: sentence: Sentence
    :rtype: float
    """

    return reduce(lambda x, y: numpy.multiply(x, y),
                  sentence.values())

def sv_kintsch(sentence, model=MODEL):
    """Returns an additive sentence vector that is extended by an
    additional word vector. That word vector is the vector of the
    closest word of the predicate, as determined by NLTK.

    :params: sentence: The sentence whose sentence vector to calculate
    :type: sentence: Sentence
    :rtype: float
    """

    tagged_words = nltk.pos_tag(str(sentence).split())
    sentence_vector = reduce(lambda x, y: x + y, sentence.values())

    for word in tagged_words:
        if word[1] in "VB VBD VBG VBN VBP VBZ".split():
            # Look for the most similar word in model
            closest_word = model.most_similar(word[0])[0][0]
            sentence_vector += model[closest_word]

    return sentence_vector

def similarity(sentence1, sentence2, sv_function=sv_add):
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

def compare(sentence1, sentence2, sv_function=sv_add):
    """
    """

    def get_cluster(sentence):
        """
        """

        print('')
        print("Choose the best cluster for the sentence")
        print('>', str(sentence))
        print("or type `0' to re-roll clusters:")

        while True:
            for i, cluster in enumerate(sentence.clusters):
                print(i+1, cluster)

            print("0 re-roll clusters")
            input_cluster = int(input())

            if input_cluster == 0:
                sentence.clusterize()
                continue

            try:
                return sentence.clusters[input_cluster-1]
            except IndexError:
                continue

    print("Comparing:\n", ">", str(sentence1), "\n", ">", str(sentence2))

    first_cluster = get_cluster(sentence1)
    second_cluster = get_cluster(sentence2)

    first_sentence = Sentence(" ".join(s for s in first_cluster))
    second_sentence = Sentence(" ".join(s for s in second_cluster))

    return 1 - scipy.spatial.distance.cosine(sv_function(first_sentence),
                                             sv_function(second_sentence))


def evaluate_ml_corpus(corpus=ML_CORPUS, participant="participant1"):
    """
    """

    similarities = []

    # `v' is a triple of a sentence-sentence-similarity of Mitchell's
    # and Lapata's evaluation corpus, so v[0], v[1], etc. is used to
    # access each individual item.
    for i, v in enumerate(corpus[participant]):
        add_cosine = similarity(v[0], v[1])
        multiply_cosine = similarity(v[0], v[1], sv_multiply)
        weightadd_cosine = similarity(v[0], v[1], sv_weightadd)
        kintsch_cosine = similarity(v[0], v[1], sv_kintsch)

        similarities.append([v[0], v[1], v[2],
                             cosine_to_integer(add_cosine),
                             cosine_to_integer(multiply_cosine),
                             cosine_to_integer(weightadd_cosine),
                             cosine_to_integer(kintsch_cosine)])

        # Every 20 lines (or when the corpus reached its last line),
        # print the sentence similarities
        if len(similarities) == 20 or i == len(corpus[participant]) - 1:
            print(tabulate(similarities,
                           headers=["1st Sentence", "2nd Sentence", "M&L",
                                    "+", "W.+", "*", "K."]))
            print("")

            # Flush the buffer for the next 20 lines
            similarities = []
