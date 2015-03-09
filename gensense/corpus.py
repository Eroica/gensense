
class EvaluationCorpus(list):
    def __init__(self):

        super(EvaluationCorpus, self).__init__()

        with open("share/share.html") as f:
            for line in f:
                eval_line = line.split()[3:]
                self.append([" ".join(eval_line[0:2]),
                             " ".join(eval_line[2:4]),
                             eval_line[-1]])
                # self.append([x[4:] for x in line])

        self.evaluation_file = "share/share.html"

    def iterator(self):
        with open(self.evaluation_file) as f:
            for line in f:
                yield line