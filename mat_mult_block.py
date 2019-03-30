from mpi4py import MPI
import numpy as np
from matmult_funcs import *

world = MPI.COMM_WORLD
rank = world.Get_rank()
nProcs = world.Get_size()

# Initialisation
if (rank==0):
    # init_matrices is defined such that it takes two arrays for size of A and B respectively and two more arguments for lowest and highest random number to be generated. Default for low is 0 and high is 1
    sizeA = [64,160]
    sizeB = [160,16]
    A,B = init_matrices(sizeA,sizeB)
    iProcs,jProcs = factor_procs(nProcs,sizeA,sizeB)
    print(iProcs,jProcs)

# Decomposition
# iProcs = world.bcast(iProcs,root=0)
# jProcs = world.bcast(jProcs,root=0)

# Communication based on decomposition

# Calculation

# Communication to master

print("Done",rank)
