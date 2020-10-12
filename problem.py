import mlrose_hiive as mlrose
class Problem:
    def __init__(self, problem: mlrose.DiscreteOpt, prob_name: str):
        self.problem = problem
        self.prob_name = prob_name