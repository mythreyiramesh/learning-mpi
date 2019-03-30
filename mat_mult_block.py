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
    sizeA = [16,10]
    sizeB = [10,16]
    A,B = init_matrices(sizeA,sizeB)
    C = np.zeros((sizeA[0],sizeB[1]),dtype='d')
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
# print(local_A_rows,rank)
#
# if rank == 0:
#     print(B)
# else:
# print(local_B_cols,rank)

# Calculation
local_C_block = np.dot(local_A_rows,local_B_cols)

# Communication to master

# Each processor with rank "rank" is actually I,J th processor in the processor grid
# I = int(rank/jProcs)
# J = rank%jProcs
# Processor (I,J) will communicate block (I,J) to master. The start and end indices are given as
# C_I = I*sizeC[0] to (I+1)*sizeC[0]
# C_J = J*sizeC[1] to (J+1)*sizeC[1]
# proc_grid_i = int(rank/jProcs)
# proc_grid_j = rank%jProcs
# proc_grid_id = proc_grid_i*jProcs+proc_grid_j this is rank!
if rank != 0:
    world.Send([local_C_block,MPI.DOUBLE],dest=0,tag=300+rank)
else:
    C[rank*blockSize[0]:(rank+1)*blockSize[0],rank*blockSize[1]:(rank+1)*blockSize[1]] = local_C_block
    for i_proc in range(0,iProcs):
        for j_proc in range(0,jProcs):
            proc_id = i_proc * jProcs + j_proc
            if proc_id != 0:
                recv_tag_C = 300 + proc_id
                world.Recv(local_C_block,source=proc_id,tag=recv_tag_C)
                C[i_proc*blockSize[0]:(i_proc+1)*blockSize[0],j_proc*blockSize[1]:(j_proc+1)*blockSize[1]] = local_C_block
    # print(C)

if rank == 0:
    C_act = np.dot(A,B)
    # print(C_act,"actual")
    print(np.amax(np.abs(C-C_act)))
print("Done",rank)

# Future: Add compatibility for non-equal last bits to incorporate 3 processors, say
