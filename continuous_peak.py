import mlrose_hiive as mlrose
from problem import Problem

class Continous_Peak(Problem):
    def __init__(self, t_pct: float, prob_len: int):
        self.t_pct = t_pct
        self.prob_len = prob_len
        self.fitness = mlrose.ContinuousPeaks(t_pct=t_pct)
        problem = mlrose.DiscreteOpt(length=prob_len, fitness_fn=self.fitness, maximize=True, max_val=2)
        super().__init__(problem=problem, prob_name='continuous_peak')