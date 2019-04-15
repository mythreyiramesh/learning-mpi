import numpy as np
from math import ceil,sqrt

def T_1Dinit(size):
    # let the starting temperature profile be a sin curve
    # x_grid = np.arange(size[0],dtype='d')
    # y_grid = np.arange(size[1],dtype='d')
    T = (np.random.rand(size))
    return T

def write_to_file_1d(T,step):
    size = np.shape(T)[0]
    line = "T(" + str(step) +",:) = ["
    for i in range(1,size,100):
        line = line + str(float(T[i])) + " "
    line = line + "];\n"
    f = file("Tout.m","a")
    f.writelines(line)
    f.close()
    step = step+1
    return step

def clear_output_file():
    f = file("Tout.m","w")
    line = ""
    f.close()
