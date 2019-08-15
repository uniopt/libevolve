# -*- coding: utf-8 -*-

import pytest
from libevolve.common import *
from libevolve.ga import GeneticAlgorithm,NSGA_II



def test_ga_constructor():
    """
    test the constructor function of the genetic algorithms module
    """
    ga = GeneticAlgorithm()
    ga = GeneticAlgorithm(population_size=10, nb_generations=15, mutation_size=1, mutation_probability=0.9)

    assert ga.population_size == 10, "Population size not assigned properly"
    assert ga.nb_generations == 15, "Population size not assigned properly"
    assert ga.mutation_size == 1, "Population size not assigned properly"
    assert ga.mutation_probability == 0.9, "Population size not assigned properly"


def test_ga_evolve_exceptions():
    """

    """
    a = EvoIntParam(name="a", min_val=0, max_val=10)
    b = EvoFloatParam(name="b", min_val=10, max_val=20, step=0.5)
    c = EvoIntParam(name="c")
    d = EvoIntParam(name="d")
    e = EvoIntParam(name="e")
    def objective_fn(a,b, c):
        res = (b-c)
        return res
    ga = GeneticAlgorithm()
    best_score, best_params, history = ga.evolve(parameters=[c,a, b],
                                                 fitness_function=objective_fn, objective_weights=[-1, 1])
    best_params = best_score[1,:]
    best_score = best_score[0]

    def objective_fn_1(d,e):
        res = d + e
        return res

    nsga = NSGA_II()
    best_solution, best_score, history =nsga.evolve(parameters=[d,c,e,a,b],fitness_function=[objective_fn_1],objective_weights=[1,1,1,1,1])
