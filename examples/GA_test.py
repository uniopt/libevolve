from libevolve.common._base import *
from libevolve.deap.ga._base import *

import numpy as np
import random,string


def fitness_func(v1, v2, v3):
    """ Sample objective function for evolutionary algorithm

    :param v1: int
        int value of param 1
    :param v2: int
        int value of param 2
    :param v3: character
        char value of param 3

    :return:
    float : objective value

    """
    v1_rnd = np.sum(-np.abs(v1 - np.random.randint(50, 80, [5, ])))
    v2_rnd = np.sum(-np.abs(v1 - np.random.randint(15, 54, [5, ])))

    v3 = ord(v3)
    return -(v1 + v2 - v3) ** 2 + v1_rnd + v2_rnd


def eval2(part):
    return sum(part),


v1 = EvoFloatParam("x1", min_val=-3, max_val=35)
v2 = EvoFloatParam("x2", min_val=-1, max_val=35)


# v3_range = [c for c in string.ascii_lowercase]
# v1 = EvoIntParam("v1",min_val=0,max_val=100)
# v2 = EvoIntParam("v2",min_val=0,max_val=100)
# v3 = EvoParam("v3",v3_range)

params = [v1, v2]

c_GA = GeneticAlgorithm(population_size=5, nb_generations=10,verbose=True)
value, ind, hist = c_GA.evolve(parameters=params, fitness_function=eval2, objective_weights=(1.0,))
print(ind, value)

print(len(hist))



