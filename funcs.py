import numpy as np
from math import sqrt,floor

def init_input_matrices(size,low,up):
    A = (np.random.rand(size,size)*(up-low)) + low;
    B = (np.random.rand(size,size)*(up-low)) + low;
    return A,B

def get_procs(C_size,nprocs):
    nprocs_root = floor(sqrt(nprocs))
    iProcs = int(nprocs_root)
    jProcs = int(nprocs_root)
    return iProcs,jProcs
