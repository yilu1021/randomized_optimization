import mlrose_hiive as mlrose
import numpy as np
import json
from problem import Problem
from typing import List
import matplotlib.pyplot as plt
random_state = 1

class Solver:
    def __init__(self, solver, solver_name: str, problem: Problem, func_eval_multiplier: int):
        self.solver = solver
        self.solver_name = solver_name
        self.problem = problem
        self.best_fitness_curve = None
        self.best_state = None
        self.best_fitness = None
        self.func_eval_multiplier = func_eval_multiplier

    def solve(self):
        pass

    def save_result(self, best_state, best_fitness, best_fitness_curve):
        self.best_state = best_state
        self.best_fitness = best_fitness
        self.best_fitness_curve = best_fitness_curve

        print(self.problem.prob_name, self.solver_name, best_fitness)
        data = {'prob_name': self.problem.prob_name, 'solver_name': self.solver_name, 'best_fitness': float(best_fitness),
                'best_state': best_state.tolist(), 'best_fitness_curve': best_fitness_curve.tolist()}
        with open('output/' + self.problem.prob_name + "_" + self.solver_name + '.json', 'w') as outfile:
            json.dump(data, outfile)
        pass

class RandomizedHillClimbing(Solver):
    def __init__(self, max_attempts: int, max_iters: int, restarts: int, problem: Problem):
        super().__init__(solver=mlrose.random_hill_climb, solver_name='RHC', problem=problem,
                         func_eval_multiplier=1)
        self.max_attempts = max_attempts
        self.max_iters = max_iters
        self.restarts = restarts

    def solve(self):
        best_state, best_fitness, best_fitness_curve = self.solver(problem=self.problem.problem,
                                                               max_attempts=self.max_attempts,
                                                                   max_iters=self.max_iters,
                                                                   restarts=self.restarts,
                                                                   curve=True, random_state=random_state)
        self.save_result(best_state, best_fitness, best_fitness_curve)

class SimulatedAnnealing(Solver):
    def __init__(self, schedule, max_attempts: int, max_iters: int, problem: Problem):
        super().__init__(solver=mlrose.simulated_annealing, solver_name='SA', problem=problem, func_eval_multiplier=1)
        self.schedule = schedule
        self.max_attempts = max_attempts
        self.max_iters = max_iters

    def solve(self):
        best_state, best_fitness, best_fitness_curve = self.solver(problem=self.problem.problem, schedule=self.schedule,
                                                                   max_attempts=self.max_attempts,
                                                                   max_iters=self.max_iters, curve=True, random_state=random_state)
        self.save_result(best_state, best_fitness, best_fitness_curve)

class GeneticAlgo(Solver):
    def __init__(self, pop_size: int, mutation_prob: float, max_attempts: int, max_iters: int, problem: Problem):
        super().__init__(solver=mlrose.genetic_alg, solver_name='GA', problem=problem, func_eval_multiplier=pop_size)
        self.pop_size = pop_size
        self.mutation_prob = mutation_prob
        self.max_attempts = max_attempts
        self.max_iters = max_iters

    def solve(self):
        best_state, best_fitness, best_fitness_curve = self.solver(problem=self.problem.problem, pop_size=self.pop_size,
                                                                   mutation_prob=self.mutation_prob,
                    max_attempts=self.max_attempts, max_iters=self.max_iters, curve=True, random_state=random_state)
        self.save_result(best_state, best_fitness, best_fitness_curve)

class MIMIC(Solver):
    def __init__(self, pop_size: int, keep_pct: float, max_attempts: int, max_iters: int, problem: Problem):
        super().__init__(solver=mlrose.mimic, solver_name='MIMIC', problem=problem, func_eval_multiplier=pop_size)
        self.pop_size = pop_size
        self.keep_pct = keep_pct
        self.max_attempts = max_attempts
        self.max_iters = max_iters

    def solve(self):
        best_state, best_fitness, best_fitness_curve = self.solver(problem=self.problem.problem, pop_size=self.pop_size,
                                                                   keep_pct=self.keep_pct,
                    max_attempts=self.max_attempts, max_iters=self.max_iters, curve=True, random_state=random_state)
        self.save_result(best_state, best_fitness, best_fitness_curve)


def plot(maximize: bool, title: str, ros: List[Solver]):
    title1 = title + '_iterations'
    plt.title(title1)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Score')
    for ro in ros:
        y_vals = ro.best_fitness_curve
        if maximize is False:
            y_vals = ro.best_fitness_curve * -1
        plt.plot(range(0, ro.best_fitness_curve.shape[0]), y_vals, label=ro.solver_name)
    plt.legend(loc='best')
    plt.savefig('output_part1/' + title1 + '.png')
    plt.close()

    title2 = title + '_function_evaluations'
    plt.title(title2)
    plt.xlabel('Function Evaluations')
    plt.ylabel('Fitness Score')
    for ro in ros:
        y_vals = ro.best_fitness_curve
        if maximize is False:
            y_vals = ro.best_fitness_curve * -1
        plt.semilogx(np.arange(ro.best_fitness_curve.shape[0]) * ro.func_eval_multiplier, y_vals, label=ro.solver_name)
    plt.legend(loc='best')
    plt.savefig('output_part1/' + title2 + '.png')
    plt.close()
