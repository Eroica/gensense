"""This module is used to access Mitchell's and Lapata's evaluations of
sentence similarities as done in their paper ``Vector-based Models of
Semantic Composition''

"""

from sentence import Sentence

class EvaluationCorpus(list):
    """This class holds every sentence pair of Mitchell's and Lapata's
    evaluation file. Each pair is represented by a list of 3 elements:
    The two sentences being compared, and their similarity grade (an
    integer between 1 and 7).

    This is a huge data structure, and as such should only be used by
    stateless iterators, to avoid too much memory allocation and long
    waiting times.

    Examples on how to use this data structure is found in
    `gensense.evaluate'.

    """

    def __init__(self, evaluation_file="share/share.html"):
        """Iterates over the evaluation file, and creates a list of two
        `Sentence' objects for the two sentences found in each line,
        then appending them to `self'. For each list of `Sentence'
        pairs, their similarity score (as found in the file) is also
        appended at the end.
        """

        super(EvaluationCorpus, self).__init__()

        with open("share/share.html") as f:
            # Skip the first line, as it only contains some information
            # on how each line is structured
            next(f)

            for line in f:
                eval_line = line.split()[3:]

                sentence_left = Sentence(" ".join(eval_line[0:2]))
                sentence_right = Sentence(" ".join(eval_line[2:4]))

                self.append([sentence_left,
                             sentence_right,
                             eval_line[-1]])
