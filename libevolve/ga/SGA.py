import random

from ._base import GeneticAlgorithm
from deap import base, algorithms, creator, tools
from libevolve.libevolve.common._base import EvoCategoricalParam,EvoIntParam,EvoFlagParam, EvoFloatParam

class SGA(GeneticAlgorithm):

    def __init__(self,population_size,parameters,fitness_function,ngen,objective_weights,fitness_type,k=5,tournsize=5,mutpb=0.1,cxpb=0.1):
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
        super().__init__(population_size,parameters,fitness_function,ngen,objective_weights,fitness_type,mutpb=mutpb,cxpb=cxpb)

        self.population=self.Evaluate(fitness_function,self.population)
        offspring = []
        parents = []
        for g in range(self.ngen):
            parents = self.Select(fitness_function,k,tournsize)
            offspring = self.Cross_Over(parents)
            offspring = self.Mutation(offspring)
            offspring=self.Evaluate(fitness_function,offspring)
            self.population = offspring

    def get_result(self):
        result = []
        for ind in self.population:
            l1 = []
            l1.append(ind["result"])
            for key,value in ind.items():
                if key != "result":
                    l1.append(value._val)
            result.append(l1)
        return result

    def Select(self,fitness_function,k,tournsize):
        """
        Tournament Selection:

        In K-Way tournament selection, we select K individuals from the population at random and select the best out of these to become a parent.
        The same process is repeated for selecting the next parent.
        Tournament Selection is also extremely popular in literature as it can even work with negative fitness values.

        :param fitness_function:The Objective function
        :param k: The number of individuals to select
        :param tournsize: The number of individuals participating in each tournament.
        :return: selected parents
        """
        selected = []
        for i in range(k):
            parents = []
            for j in range(tournsize):
                parents.append(random.choice(self.population))
            mx = -1000000000
            for ind in parents:
                val = ind["result"]
                if val>mx :
                    mx = val
                    ind_mx = ind
            selected.append(ind_mx)

        return selected

    def Cross_Over(self,offspring):
        """
            One Point Crossover:
            In this one-point crossover, a random crossover point is selected and the tails of its two parents
            are swapped to get new off-springs.

        :param offspring:
        :return offspring:
        """

        for i in range(1,len(offspring),2):
            if random.random()<self.cxpb:
                ind1=list(offspring[i-1].values())
                ind2 = list(offspring[i].values())
                ind_1,ind_2= tools.cxOnePoint(ind1,ind2)
                dict = {}
                iter = 0

                for key ,value in offspring[i-1].items():
                    if "int" in key:
                        ind = EvoIntParam(name=key,max_val=ind_1[iter].get_max_value(),min_val=ind_1[iter].get_min_value())
                        ind._val = ind_1[iter].get_rand_value()
                        dict[key] = ind
                    elif "Categorical" in key or "char" in key :
                        ind = EvoCategoricalParam(name=key,categories=ind_1[iter].categories)
                        ind._val = ind_1[iter].get_rand_value()
                        dict[key] = ind
                    elif "float" in key:
                        ind =  EvoFloatParam(name=key,max_val=ind_1[iter].get_max_value(),min_val=ind_1[iter].get_min_value(),step=ind_1[iter].get_step_value())
                        ind._val = ind_1[iter].get_rand_value()
                        dict[key] = ind
                    elif "binary" in key:
                        ind = EvoFlagParam(name=key)
                        ind._val = ind.get_rand_value()
                        dict[key] = ind
                    iter+=1

                offspring[i-1]=dict

                dict = {}
                iter = 0
                for key ,value in offspring[i].items():
                    if "int" in key:
                        ind = EvoIntParam(name=key,max_val=ind_2[iter].get_max_value(), min_val=ind_2[iter].get_min_value())
                        ind._val = ind_2[iter].get_rand_value()
                        dict[key] = ind
                    elif "Categorical" in key or "char" in key:
                        ind = EvoCategoricalParam(name=key,categories=ind_2[iter].categories)
                        ind._val = ind_2[iter].get_rand_value()
                        dict[key] = ind
                    elif "float" in key:
                        ind = EvoFloatParam(name=key,max_val=ind_2[iter].get_max_value(), min_val=ind_2[iter].get_min_value(),
                                            step=ind_2[iter].get_step_value())
                        ind._val = ind_2[iter].get_rand_value()
                        dict[key] = ind
                    elif "binary" in key:
                        ind = EvoFlagParam(name=key)
                        ind._val = ind.get_rand_value()
                        dict[key] = ind
                    iter+=1
                offspring[i] = dict
        return  offspring


    def Mutation(self,offspring):
        """
        Random Resetting:

        Random Resetting is an extension of the bit flip for the integer representation.
        In this, a random value from the set of permissible values is assigned to a randomly chosen gene.

        :param offspring:
        :return: offspring:
        """
        for i in range(0,len(offspring)):
            if random.random()<self.mutpb:
                rn = random.choice(list(offspring[i].keys()))
                if rn != "result" :
                    offspring[i][rn]._val=offspring[i][rn].get_rand_value()


        return offspring

