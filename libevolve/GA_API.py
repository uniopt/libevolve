import itertools
import string
import numpy as np
from libevolve.libevolve.ga.SGA  import SGA

def objective_fn(param,w):
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

    v1_rnd = 20 #np.sum(-np.abs(v1 - np.random.randint(50, 80, [5, ])))

    v2_rnd = 10 #np.sum(-np.abs(v1 - np.random.randint(15, 54, [5, ])))
    v1,v2,v3,v4,v5= param
    # print(-(v1 + v2 - ord(v3)) ** 2 + v1_rnd + v2_rnd)
    return -(v1 + v2 - ord(v3)) ** 2 + v1_rnd + v2_rnd +v4


a_list = list(range(100))

b_list = list(range(100))

c_list = [c for c in string.ascii_lowercase]
d_list = list(np.arange(0, 20.0, 1.0))

inputs = list(itertools.product(a_list, b_list, c_list,d_list))

outputs = np.array([objective_fn((a, b, c, d),(1.0,)) for a, b, c, d in inputs])

sort_indices = np.argsort(outputs)[::-1]

sorted_outputs = outputs[sort_indices]

sorted_inputs = np.array(inputs)[sort_indices]

best_solution = sorted_inputs[0]

print("Best solution by BF: %s" % best_solution,sorted_outputs[0])

dict = {"int":(0,100),"int 1":(0,100),"char":['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'],"float":(0.0,20.0,1.0)}

test = SGA(100,dict,objective_fn,100,(1.0,0),"Max",k=10,tournsize=10,mutpb=0.8,cxpb=0.6)
test = test.get_result()
test.sort(reverse=True)
print("Best solution by GA: %s" % test[0])