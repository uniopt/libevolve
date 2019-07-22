from libevolve.common._base import *
from libevolve.swarm.SwarmOp import *
from numpy import exp


def eval(part):
    x= part[0]
    y= part[1]
    return (3*( (1-x)**2 ) * exp( -(x**2)-(y+1)**2 ) - 10*(x/5 - (x**3) -(y**5)) * exp(-(x**2) - (y**2)) - (1/3)*exp(-((x+1)**2) -(y**2))),


x1 = EvoFloatParam("x1",min_val=-3,max_val=20)
x2 = EvoFloatParam("x2",min_val=-1,max_val=15)
parameters = [x1,x2]
speedranges = [(-2,2),(-2,2)]

swarm = SwarmOptimization(nb_generations=50)

pop , log, best = swarm.evolve(parameters=parameters, speedranges=speedranges,
                               fitness_function=eval,objective_weights=(1.0,))

print(pop)
print(best)
