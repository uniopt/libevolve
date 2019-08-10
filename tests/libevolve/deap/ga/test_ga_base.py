# -*- coding: utf-8 -*-

import pytest
from libevolve.common import *
from libevolve.deap.ga import GeneticAlgorithm


def test_ga_constructor():
    """
    test the constructor function of the genetic algorithms module
    """
    ga = GeneticAlgorithm()
    ga = GeneticAlgorithm(population_size=10, nb_generations=15, crossover_probability=0.5, mutation_probability=0.9)

    assert ga.population_size == 10, "Population size not assigned properly"
    assert ga.nb_generations == 15, "Population size not assigned properly"
    assert ga.crossover_probability == 0.5, "Population size not assigned properly"
    assert ga.mutation_probability == 0.9, "Population size not assigned properly"


def test_ga_evolve_exceptions():
    """

    """
    a = EvoIntParam(name="a", min_val=0, max_val=10)
    b = EvoFloatParam(name="b", min_val=10, max_val=20, step=0.5)

    def objective_fn(ind):
        c = ind[0]
        d = ind[1]
        return 0.1 * (c-d),

    ga = GeneticAlgorithm()
    best_score, best_params, history = ga.evolve(parameters=[a, b],
                                                 fitness_function=objective_fn,
                                                 objective_weights=[-1, 1])