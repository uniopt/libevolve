# -*- coding: utf-8 -*-

import pytest
from libevolve.common import *
from libevolve.deap.swarm.SwarmOp import SwarmOptimization


def test_ga_constructor():
    """
    test the constructor function of the genetic algorithms module
    """
    swarm = SwarmOptimization()
    swarm = SwarmOptimization(population_size=10, nb_generations=15)

    assert swarm.population_size == 10, "Population size not assigned properly"
    assert swarm.nb_generations == 15, "Number of generations not assigned properly"


def test_ga_evolve_exceptions():
    """

    """
    a = EvoIntParam(name="a", min_val=0, max_val=10)
    b = EvoFloatParam(name="b", min_val=10, max_val=20, step=0.5)

    def objective_fn(part):
        c = part[0]
        d = part[1]
        return 0.1 * (c-d),

    speedranges = [(-2, 2), (-2, 2)]
    swarm = SwarmOptimization()
    best_score, best_params, history = swarm.evolve(parameters=[a, b], speedranges=speedranges,
                                                    fitness_function=objective_fn,
                                                    objective_weights=[-1, 1])
    assert type(best_score) == tuple
