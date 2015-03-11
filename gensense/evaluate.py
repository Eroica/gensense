from corpus import EvaluationCorpus

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
    return int(math.ceil(grade * 7))