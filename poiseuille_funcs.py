import numpy as np
from math import ceil,sqrt

def U_2Dinit(size):
    # let the starting temperature profile be a sin curve
    # x_grid = np.arange(size[0],dtype='d')
    # y_grid = np.arange(size[1],dtype='d')
    U = (np.random.rand(size[0],size[1]))
    return U

def write_to_file_2d(T,step):
    f = file("Uout_couette.m","a")
    height = np.shape(T)[0]
    width = np.shape(T)[1]
    line = "U(" + str(step) +", :,:) = ["
    for i in range(1,height-1):
        for j in range(1,width-1):
            line = line + str(float(T[i,j])) + " "
        line = line + ";\n"
    line = line + "];\n\n"
    f.writelines(line)
    f.close()
    step = step+1
    return step

def factor_procs(nProcs,size):
    flag = 0 # Keeping track of larger dimension
    if size[0] > size[1]:
        aspect_ratio = size[0]/size[1]
    else:
        aspect_ratio = size[1]/size[0]
        flag = 1
    f1 = ceil(sqrt(float(nProcs)/aspect_ratio))
    while (nProcs%f1 != 0):
        f1 = f1-1
    f2 = nProcs/f1
    if flag:
        return int(f1),int(f2)
    else:
        return int(f2),int(f1)

def clear_output_file():
    f = file("Uout_couette.m","w")
    line = ""
    f.close()
