from mpi4py import MPI
import numpy as np
from matmult_funcs import *

world = MPI.COMM_WORLD
rank = world.Get_rank()
nProcs = world.Get_size()

# Default Values
iProcs = 1
jProcs = 1
# sizeA = np.zeros((2,1),dtype='d')
# sizeB = np.zeros((2,1),dtype='d')
sizeA = [0, 0]
sizeB = [0, 0]

# Global Initialisation
if (rank==0):
    # init_matrices is defined such that it takes two arrays for size of A and B respectively and two more arguments for lowest and highest random number to be generated. Default for low is 0 and high is 1
    sizeA = [8,5]
    sizeB = [5,2]
    A,B = init_matrices(sizeA,sizeB)
    iProcs,jProcs = factor_procs(nProcs,sizeA,sizeB)
    # print(iProcs,jProcs)

# Communication of the decomposition
sizeA = world.bcast(sizeA,root=0)
sizeB = world.bcast(sizeB,root=0)
iProcs = world.bcast(iProcs,root=0)
jProcs = world.bcast(jProcs,root=0)

# print(sizeA,sizeB,iProcs,jProcs,rank)

# Local Initialisation
sizeC = [sizeA[0],sizeB[1]] # not necessary, but present for clarity
blockSize = [sizeC[0]/iProcs,sizeC[1]/jProcs]
blockLength = sizeA[1] # by now, it will be the same as sizeB[0]
# Here, blockSize stores the number of rows of A and cols of B to be received and blockLength stores the length of the block

local_A_rows = np.zeros((blockSize[0],blockLength),dtype='d')
local_B_cols = np.zeros((blockLength,blockSize[1]),dtype='d')
local_C_block = np.zeros((blockSize[0],blockSize[1]),dtype='d')

# print(blockSize,rank)

# Communication based on decomposition

# Rank 0 is behaving as the master and it communicates various parts of matrix A and B to the different processors, keeping aside some for itself
if (rank==0):
    i_indices = np.linspace(0,sizeC[0],iProcs+1)
    j_indices = np.linspace(0,sizeC[1],jProcs+1)
    print(i_indices)
    print(j_indices)

    for i_proc in range(0,iProcs):
        for j_proc in range(0,jProcs):
            proc_id = i_proc * jProcs + j_proc
            if proc_id == 0:
                local_A_rows = A[i_indices[i_proc]:i_indices[i_proc+1],:]
                local_B_cols = B[:,j_indices[j_proc]:j_indices[j_proc+1]]
            else:
                send_tag_A = 100+proc_id
                send_tag_B = 200+proc_id
                world.Send([A[i_indices[i_proc]:i_indices[i_proc+1],:],MPI.DOUBLE],dest=proc_id,tag=send_tag_A)
                world.Send([B[:,j_indices[j_proc]:j_indices[j_proc+1]],MPI.DOUBLE],dest=proc_id,tag=send_tag_B)
else:
    world.Recv(local_A_rows,source=0,tag=100+rank)
    world.Recv(local_B_cols,source=0,tag=200+rank)

# if rank == 0:
#     print(A)
# else:
#     print(local_A_rows,rank)
#
# if rank == 0:
#     print(B)
# else:
#     print(local_B_cols,rank)

# Calculation

# Communication to master

print("Done",rank)
