"""This module is used to

"""

class EvaluationCorpus(list):
    """


    """

    def __init__(self):
        """
        """

        super(EvaluationCorpus, self).__init__()

        self.evaluation_grades = []

        with open("share/share.html") as f:
            for line in f:
                eval_line = line.split()[3:]
                self.evaluation_grades = eval_line[-1]

                self.append([" ".join(eval_line[0:2]),
                             " ".join(eval_line[2:4])])

        self.evaluation_file = "share/share.html"