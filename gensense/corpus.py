
class EvaluationCorpus(list):
    def __init__(self):

        super(EvaluationCorpus, self).__init__()

        with open("share/share.html") as f:
            for line in f:
                eval_line = line.split()[3:]
                self.append([" ".join(eval_line[0:2]),
                             " ".join(eval_line[2:4]),
                             eval_line[-1]])

        self.evaluation_file = "share/share.html"

        # Since ... original evaluation used numbers from 1 to 7 to rate
        # the similarity between two sentences, this list is used to
        # map values from 0.0 to 1.0 to values from 1 to 7.
        # The same list procedurally generated could look like this:
        #
        #   self.grads = map(lambda x: 1 - x * (1.0/7),
        #                    [x for x in range(0, 8)])
        #
        self.grads = [1.0, 6.0/7, 5.0/7, 4.0/7,
                      3.0/7, 2.0/7, 1.0/7, 0.0]

    def iterator(self):
        with open(self.evaluation_file) as f:
            for line in f:
                yield line