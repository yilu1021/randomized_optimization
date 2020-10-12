import numpy as np
from solver import RandomizedHillClimbing, SimulatedAnnealing, GeneticAlgo, MIMIC, plot
from continuous_peak import Continous_Peak
from four_peak import Four_Peak
from traveling_salesman import Traveling_Salesman
from mlrose_hiive.algorithms.decay import GeomDecay
from nn import NN

random_state = 1
np.random.seed(random_state)

def exp_continous_peak():
    prob_len = 50
    restarts = 10
    max_iters = 1000
    max_attempts = 1000

    def gen_prob():
        return Continous_Peak(t_pct=0.1, prob_len=prob_len)

    rhc=RandomizedHillClimbing(max_attempts=int(max_attempts), max_iters=int(max_iters),
                                    restarts=restarts, problem=gen_prob())
    rhc.solve()

    sa=SimulatedAnnealing(schedule=GeomDecay(), max_attempts=max_attempts, max_iters=max_iters, problem=gen_prob())
    sa.solve()

    ga=GeneticAlgo(pop_size=50, mutation_prob=0.1, max_attempts=max_attempts, max_iters=max_iters, problem=gen_prob())
    ga.solve()

    mimic = MIMIC(pop_size=100, keep_pct=0.2, max_attempts=max_attempts, max_iters=max_iters, problem=gen_prob())
    mimic.solve()

    plot(maximize=True, title='continuous_peak', ros=[rhc, sa, ga, mimic])


def exp_continous_peak_rhc_restarts():
    prob_len = 100
    max_iters = 1000
    max_attempts = 1000

    def gen_prob():
        return Continous_Peak(t_pct=0.1, prob_len=prob_len)

    # restarts_max = 100
    restarts_max = [1, 10, 20, 30, 40, 50]

    ros = []
    for restarts in restarts_max:
        rhc = RandomizedHillClimbing(max_attempts=int(max_attempts), max_iters=int(max_iters),
                                     restarts=restarts, problem=gen_prob())
        rhc.problem.prob_name = 'Allowed Restarts: ' + str(restarts)
        rhc.solve()
        ros.append(rhc)

    plot(maximize=True, title='continuous_peak_rhc_restarts', ros=ros)


def exp_four_peak():

    prob_len = 120
    restarts=10

    def gen_prob():
        return Four_Peak(t_pct=0.1, prob_len=prob_len)

    rhc = RandomizedHillClimbing(max_attempts=7500, max_iters=7500, restarts=restarts, problem=gen_prob())
    rhc.solve()

    sa = SimulatedAnnealing(schedule=GeomDecay(), max_attempts=7500, max_iters=7500, problem=gen_prob())
    sa.solve()

    ga = GeneticAlgo(pop_size=500, mutation_prob=0.1, max_attempts=15, max_iters=15, problem=gen_prob())
    ga.solve()

    mimic = MIMIC(pop_size=500, keep_pct=0.1, max_attempts=15, max_iters=15, problem=gen_prob())
    mimic.solve()

    plot(maximize=True, title='four_peak', ros=[rhc, sa, ga, mimic])

def exp_traveling_salesperson():
    cities_num = 50
    x_coords = np.random.choice(cities_num * 2, cities_num)
    y_coords = np.random.choice(cities_num * 2, cities_num)
    coords_list = []

    max_iters = 1000
    max_attemps = 1000

    for n in range(0, cities_num):
        coords_list.append((x_coords[n], y_coords[n]))

    def gen_prob():
        return Traveling_Salesman(coords_list=coords_list)

    restarts = 100
    # restarts_max = 100
    # for restarts in range(restarts_max):
    #     problem = gen_prob()
    #     problem.prob_name += ' restart: ' + str(restarts)
    #     RandomizedHillClimbing(max_attempts=max_attemps, max_iters=max_iters, restarts=restarts,
    #                            problem=problem).solve()

    rhc = RandomizedHillClimbing(max_attempts=max_attemps, max_iters=max_iters, restarts=restarts, problem=gen_prob())
    rhc.solve()

    sa = SimulatedAnnealing(schedule=GeomDecay(), max_attempts=max_attemps, max_iters=max_iters, problem=gen_prob())
    sa.solve()

    ga = GeneticAlgo(pop_size=100, mutation_prob=0.1, max_attempts=max_attemps, max_iters=max_iters, problem=gen_prob())
    ga.solve()

    mimic = MIMIC(pop_size=100, keep_pct=0.3, max_attempts=max_attemps, max_iters=max_iters, problem=gen_prob())
    mimic.solve()

    plot(maximize=False, title='traveling_salesman', ros=[rhc, sa, ga, mimic])

def exp_nn():
    nn = NN()
    nn.run()

def main():
    # GA
    exp_traveling_salesperson()
    #MIMIC
    exp_four_peak()
    #SA
    exp_continous_peak()
    #NN
    exp_nn()

    #continuous_peak_restarts
    exp_continous_peak_rhc_restarts()

if __name__ == '__main__':
    main()
