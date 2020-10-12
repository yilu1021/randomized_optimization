import mlrose_hiive as mlrose
from problem import Problem

class Traveling_Salesman(Problem):
    def __init__(self, coords_list):
        self.fitness = mlrose.TravellingSales(coords=coords_list)
        problem = mlrose.TSPOpt(length=len(coords_list), coords=coords_list, maximize=False)
        super().__init__(problem=problem, prob_name='traveling_salesman')