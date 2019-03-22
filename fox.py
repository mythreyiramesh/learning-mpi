# Algorithm to multiply two matrices using MPI

# Task Breakdown:
# 1 - Implement Fox Algorithm for a square matrix of known dimensions with "4" processors
# 2 - Modify the algorithm to incorporate any number of processors and rectagular matrices.
# 3 - Ensure the parameters for evaluation are set

# Task 1 - Implement Fox Algorithm for a square matrix of known dimensions with "4" processors

# Assumptions :
# Matrix is 8x8 with 4 processors

# Note: Initialisation can be done by one processor and later the same processor can start doing work. Another option (if you're reading from a file, use different processors to read from different parts of the file)

from mpi4py import MPI
import numpy as np
from math import floor
from funcs import *

world = MPI.COMM_WORLD
rank = world.Get_rank()
nprocs = world.Get_size() # for now, this will be 4
C_size = np.array([1,1],dtype='i'); # something arbitrary
block_size = np.array([0, 0],dtype='i') # something arbitrary
length_of_matrices = np.array([1],dtype='i'); # something arbitrary

if (rank == 0):
    # upper cap = 100, lower cap = -100
    [A,B] = init_input_matrices(4,-10,10)
    A_size = np.shape(A);
    B_size = np.shape(B);
    if A_size[1] != B_size[0]:
        print("Matrices cannot be multiplied")
        exit(1)
    else:
        length_of_matrices[0] = A_size[1]
    # C_act = A.dot(B)
    # print("Actual Ans:")
    # print(C_act)
    C_size = np.array([A_size[0],B_size[1]],dtype='i')
    # print(type(length_of_matrices))
    C = np.zeros(C_size,dtype='d')
    for i in range(1,nprocs):
        world.Send([C_size,MPI.INT],dest=i,tag=1)
        world.Send([length_of_matrices,MPI.INT],dest=i,tag=2)
else:
    world.Recv([C_size,MPI.INT],source=0)
    world.Recv([length_of_matrices,MPI.INT],source=0)

[iProcs,jProcs] = get_procs(C_size,nprocs);
# print(iProcs,jProcs)
block_size[0] = C_size[0]/iProcs
block_size[1] = C_size[1]/jProcs

# print(block_size)
no_of_steps = length_of_matrices/block_size[0] # there will be an issue here if the blocks are not square
# print(length_of_matrices,no_of_steps)

# doubt: are blocks always square?? Looks like it...

# note: since rank 0 is going to take charge of sending information to other processors, I don't have to send anything to rank 0 itself. this might change if the implementation changes

# Sending Array A to the processors
# First initialise a local A row for each processor
if (rank != 0):
    A_local_row = np.zeros((block_size[1],length_of_matrices),dtype='d')

# this can be combined into the other if condition, but keeping it here for clarity

# There will be same number of blocks in a dimension as the number of processors for that dimension. Hence, the block length will determine the start and end indices of the blocks along j.
if (rank == 0):
    # print(A)
    for i in range(0,jProcs): # is it iProcs or jProcs?
        for j in range(0,iProcs):
            proc_id = i*(iProcs)+j
            if proc_id == 0:
                continue;
            # print("Sending to",proc_id)
            rank_tag = 100 + proc_id # this is arbitrary
            start_index = (i)*block_size[0]
            end_index = C_size[0]
            if (i != jProcs-1):
                end_index = (i+1)*block_size[0]

            world.Send([A[start_index:end_index][:],MPI.DOUBLE],dest=proc_id, tag=rank_tag)
            # print('Sent A',start_index,'to',end_index,'towards rank',proc_id)

if (rank != 0):
    world.Recv([A_local_row,MPI.DOUBLE],source=0,tag=(100+rank))
    # print(rank,'received A',A_local_row,'from',0)

# Sending Array B to the processors
# First initialise a local block B for each processor
if (rank != 0):
    B_local_block = np.zeros((block_size[0],block_size[1]),dtype='d')

# this can be combined into the other if condition, but keeping it here for clarity

if (rank == 0):
    print(B)
    # print(np.shape(B))
    for i in range(0,jProcs): # is it iProcs or jProcs?
        for j in range(0,iProcs):
            proc_id = i*(iProcs)+j
            if proc_id == 0:
                continue;
            print("Sending to",proc_id)
            rank_tag = 200 + proc_id # this is arbitrary
            # doubt : since the start index and end index along the j dimension (therefore i) is always fixed as 0 and block_size[j], maybe it's just best to initialise as that or send B only from 0 to block_size[0]. NO NO BIG NO
            i_start_index = (i)*block_size[0]
            i_end_index = C_size[0]
            j_start_index = (j)*block_size[1]
            j_end_index = C_size[1]
            if (i != iProcs-1):
                i_end_index = (i+1)*block_size[0]
            if (j != jProcs-1):
                j_end_index = (j+1)*block_size[1]
            # print B[np.ix_([i_start_index,i_end_index-1],[j_start_index,j_end_index-1])]
            world.Send([B[np.ix_([i_start_index,i_end_index-1],[j_start_index,j_end_index-1])],MPI.DOUBLE],dest=proc_id, tag=rank_tag)
            print('Sent B',i_start_index,'to',i_end_index,'and',j_start_index,'to',j_end_index,'towards rank',proc_id)

if (rank != 0):
    world.Recv([B_local_block,MPI.DOUBLE],source=0,tag=(200+rank))
    print(rank,'received B',B_local_block,'from',0)

print("Done",rank)
