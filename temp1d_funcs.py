import numpy as np

def T_init(size):
    # let the starting temperature profile be a sin curve
    x_grid = np.arange(size,dtype='d')
    T = np.sin(x_grid)
    return T

def write_to_file_1d(T,step):
    size = np.shape(T)[0]
    line = "T(" + str(step) +",:) = ["
    for i in range(size):
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
