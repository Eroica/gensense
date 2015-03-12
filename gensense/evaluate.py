from __future__ import print_function
from copy import deepcopy

from corpus import EvaluationCorpus
from sentence import MODEL

import scipy.spatial
import numpy
import math
import nltk
from tabulate import tabulate

ML_CORPUS = EvaluationCorpus()

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


def sv_weightadd(sentence):
    """Returns a sentence vector where sv_i is the sum of each i-th
    component of every word vector.

    :params: sentence: The sentence whose sentence vector to calculate
    :type: sentence: Sentence
    :rtype: float
    """


    tagged_words = nltk.pos_tag(str(sentence).split())
    word_vectors = [deepcopy(vector) for vector in sentence.values()]

    for i, word in enumerate(tagged_words):
        if word[1] == 'VBP' or word[1] == 'VBN':
            word_vectors[i] *= 0.4
        elif word[1] == 'NN' or word[1] == 'PRP':
            word_vectors[i] *= 0.5
        else:
            word_vectors[i] *= 0.1

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

def sv_kintsch(sentence):
    """
    """



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

def evaluate_ml_corpus(corpus=ML_CORPUS):
    """
    """

    grades = []

    for v in corpus:
        grades.append([similarity(v[0], v[1]),
                       similarity(v[0], v[1], sv_weightadd)
                       similarity(v[0], v[1], sv_multiply)])

    print(tabulate([(v[0], v[1], v[2],
                     cosine_to_integer(grades[i][0]),
                     cosine_to_integer(grades[i][1]),
                     cosine_to_integer(grades[i][2])) for i, v in enumerate(corpus) if i < 10],

                    headers=["1st Sentence", "2nd Sentence", "M&L", "Add", "WeightAdd", "Multiply"]))
