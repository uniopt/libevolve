# -*- coding: utf-8 -*-

from abc import abstractmethod
from random import *
from .util import GeneticHistory, normalise
from ..common import *


class GeneticAlgorithm:
    """ A class for a generic genetic algorithm
    """
    def __init__(self,
                 population_size=50,
                 nb_generations=20,
                 mutation_size=1,
                 mutation_probability=0.3,
                 crossover_probability=0.5,
                 selection_size=20,
                 seed=1234,
                 verbose=0):
        """ Initialise a new instance of the `GeneticAlgorithm` class

        Parameters
        ----------
        population_size : int
            the population size
        nb_generations : int
            the number of generations
        mutation_size : int
            the number of genes to be mutated
        mutation_probability : float
            the probability of mutation for the chosen genes
        crossover_probability : float
            probability of crossover
        selection_size : int
            the size of natural selection group
        seed : int
            random seed
        verbose : int
            verbosity level

        Examples
        ----------
        >>> from libevolve.libevolve.ga import GeneticAlgorithm
        >>> ga = GeneticAlgorithm(population_size=10, nb_generations=15, mutation_size=1, mutation_probability=0.9)
        """

        self.population_size = population_size
        self.nb_generations = nb_generations
        self.mutation_size = mutation_size
        self.mutation_probability = mutation_probability
        self.crossover_probability = crossover_probability

        self.verbose = verbose
        self.seed = seed
        self.rs = Random(seed)
        self.parameters = None
        self.fitness_function = None
        self.objective_weights = None
        self.history = GeneticHistory()
    @abstractmethod
    def _mutate(self,population,size):
        """
        :param population:
        :returns: A list of varied individuals that have had changes in there genes .
        """
        offsprings = [Random.choice(population) for i in range(size)]
        for ind in offsprings:
            for i in range(1,len(ind)):
                if random.random()<self.mutation_probability:
                    ind[i]._val = ind[i].get_rand_value()

        return offsprings
    @abstractmethod
    def _crossover(self, parent1, parent2):
        """ Crossover between two parents individuals

        Parameters
        ----------
        parent1 : Individual
            first parent individual
        parent2 : Individual
            second parent individual

        Returns
        -------
        Individual
            first child individual
        Individual
            second child individual
        """
        size = len(parent1)
        for i in range(size):
            if random.random()<self.crossover_probability:
                parent1[i],parent2[i]=parent2[i],parent1[i]

        return [parent1,parent2]

    @abstractmethod
    def natural_selection(self, population,*args, **kwargs):
        """ Perform genetic natural selection

        population : list
            list of individuals
        fitness_fitness_scores : list
            list of fitness scores lists
        args : list
            other un named arguments
        kwargs : dict
            other named arguments

        Returns
        -------
        list
            list of chosen individuals
        """
        k = args
        population.sort(reverse=True)
        return population[:k]

    def evolve(self, parameters, fitness_function, objective_weights, *args, **kwargs):
        """ Perform evolution on the specified parameters and objective function

        parameters : list
            the set of evolutionary learning parameters
        fitness_function : function
            the fitness function. Expects named parameters that are equal or subset of the input parameters with the
            same names as specified in the input parameters. Must return an iterable.
        objective_weights : list
            the assigned weights to the fitness function output objective values. Positive values denote maximisation
            objective while negative values represent minimisation objective of the corresponding objective output.
        args : list
            other un named arguments
        kwargs : dict
            other named arguments

        Returns
        -------
        list
            set of best parameter values
        list
            set of fitness function scores for the best parameters
        GeneticHistory
            history of the genetic evolution

        Examples
        ----------
        >>> from libevolve.libevolve.ga import GeneticAlgorithm
        >>> from libevolve.common import *
        >>> ga = GeneticAlgorithm(population_size=10, nb_generations=15, mutation_size=1, mutation_probability=0.9)
        >>> best_solution, best_score, history = ga.evolve()
        """
        best_solution, best_score = None, 0.0

        nb_params = len(parameters)
        nb_objectives = len(objective_weights)
        param_names = [p.name for p in parameters]
        population = []

        # generate initial population
        for _ in range(self.population_size):
            ind = Individual(parameters, seed=self.rs.randint(0, 9999999999))
            population.append(ind)

        current_generation = population

        current_generation_fitness = [[] for _ in range(nb_objectives)]

        for generation_idx in range(self.nb_generations):
            """
            select
            
            """
            parents =self.natural_selection(population)
            """
            cross over
            """
            parent1_indx = Random.randint(0,len(parents))

            parent2_indx = Random.randint(0,len(parents))

            offsprings = []

            offsprings+=self._crossover(parent1_indx,parent2_indx)
            """
            mutate
            """
            size = Random.randint(1,len(offsprings))

            offsprings = self._mutate(population,size)

            population = offsprings

            current_generation = population
            for ind in current_generation:
                ind_fitness = fitness_function(**ind.key_params)
                for idx, fitness_value in enumerate(ind_fitness):
                    current_generation_fitness[idx].append(fitness_value)



        return best_solution, best_score, self.history
