# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------
# author     = "Sameh K. Mohamed"
# copyright  = "Copyright 2019, The Project"
# credits    = ["Sameh K. Mohamed"]
# license    = "MIT"
# version    = "0.0.0"
# maintainer = "Sameh K. Mohamed"
# email      = "sameh.kamaleldin@gmail.com"
# status     = "Development"
# -----------------------------------------------------------------------------------------
# Created by sameh at 2019-06-16
# -----------------------------------------------------------------------------------------
from abc import abstractmethod
from collections import Iterable
from deap import base, algorithms, creator, tools
from libevolve.libevolve.common._base import EvoCategoricalParam,EvoIntParam,EvoFlagParam, EvoFloatParam
class GeneticAlgorithm:
    """ A class for a generic genetic algorithm
    """
    def __init__(self,population_size,parameters,fitness_function,ngen,objective_weights,fitness_type,mutpb=0.1,cxpb=0.1):
        """ Initialise a new instance of the `GeneticAlgorithm` class

        Parameters
        ----------
        parameters : Iterable
            the set of evolutionary learning parameters
        fitness_function : function
            the fitness function. Expects named parameters that are equal or subset of the input parameters with the
            same names as specified in the input parameters. Must return an iterable.
        objective_weights : Iterable
            the assigned weights to the fitness function output objective values. Positive values denote maximisation
            objective while negative values represent minimisation objective of the corresponding objective output.
        """
        self.fitness_type = fitness_type
        self.population_size = population_size
        self.parameters = parameters
        self.fitness_function = fitness_function
        self.objective_weights = objective_weights
        self.ngen = ngen
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.population=self.Init_Population(self.parameters,self.population_size)





    def Init_Population(self,paramter,population_size):
        """
        Random Initialization − Populate the initial population with completely random solutions.

        :param paramter:
        :param population_size:
        :return: the population
        """
        population = []
        for i in range(population_size):
            individual = {}
            for key,value in paramter.items():

                if "int" in key:
                 mn,mx = value
                 ind = EvoIntParam(name=key,min_val=mn,max_val=mx)
                 ind._val = ind.get_rand_value()
                 individual[key] = ind

                elif "float" in key:

                 mn,mx,step = value
                 ind = EvoFloatParam(name=key,min_val=mn,max_val=mx,step=step)
                 ind._val = ind.get_rand_value()
                 individual[key] = ind

                elif "Categorical"in key or "char" in key :
                    Categorical=value
                    ind = EvoCategoricalParam(name=key,categories=Categorical)
                    ind._val = ind.get_rand_value()
                    individual[key]=ind

                elif "binary" in key:
                    ind = EvoFlagParam(name=key)
                    ind._val = ind.get_rand_value()
                    individual[key]=ind



            individual["result"]=0.0
            population.append(individual)
        return population

    @abstractmethod
    def Select(self,population,size):
        pass


    def Evaluate(self,fitness_fn,individals):
        """

        :param fitness_fn:
        :param individals:
        :return:
        """
        for ind in individals:
            l1 = []
            for key,value in ind.items():
                # print(type(value))
                if type(value) is not int and type(value) is not float:
                    l1.append(value._val)
            t1 = tuple(l1)

            ind["result"]=fitness_fn(t1,self.objective_weights)
        return individals

    @abstractmethod
    def Cross_Over(self):
        pass

    @abstractmethod
    def Mutation(self):
        pass


