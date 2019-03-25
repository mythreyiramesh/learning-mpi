# Algorithm to multiply two matrices using MPI

# Note: Initialisation can be done by one processor and later the same processor can start doing work. Another option (if you're reading from a file, use different processors to read from different parts of the file)

from mpi4py import MPI
import numpy as np
from math import floor
from helper_functions import *

world = MPI.COMM_WORLD
rank = world.Get_rank()
nProcs = world.Get_size() # for now, this will be 4
C_size = np.array([1,1],dtype='i'); # something arbitrary
A_block_size = np.array([0, 0],dtype='i') # something arbitrary
B_block_size = np.array([0, 0],dtype='i') # something arbitrary
C_block_size = np.array([0, 0],dtype='i') # something arbitrary
length_of_matrices = np.array([1],dtype='i'); # something arbitrary

if (rank == 0):
    print("Initialising matrices")
    [A,B] = init_input_matrices([50,50],[50,50])
    # print A
    # print B
    A_size = np.shape(A);
    B_size = np.shape(B);
    if A_size[1] != B_size[0]:
        print("Matrices cannot be multiplied")
        exit(1)
    else:
        length_of_matrices[0] = A_size[1]
    C_act = A.dot(B)
    C_size = np.array([A_size[0],B_size[1]],dtype='i')
    C = np.zeros(C_size,dtype='d')
    for i in range(1,nProcs):
        world.Send([C_size,MPI.INT],dest=i,tag=1)
        world.Send([length_of_matrices,MPI.INT],dest=i,tag=2)
else:
    world.Recv([C_size,MPI.INT],source=0)
    world.Recv([length_of_matrices,MPI.INT],source=0)

[iProcs,jProcs] = get_procs(C_size,nProcs);
length_of_matrices = int(length_of_matrices)
if rank == 0:
    print("Initialisation completed. Processors allocated.")

C_block_size[0] = C_size[0]/iProcs
C_block_size[1] = C_size[1]/jProcs

A_block_size[0] = C_block_size[0]
B_block_size[1] = C_block_size[1]

# # modified fox
# # By rules of matrix product, A_block_size[1] must always equal B_block_size[0]. This value can be anything! Probably must be optimised based on number of processors, length_of_matrices etc... For now, pick it to be such that at least one of the blocks are going to be square. But this is arbitrary.
#
# no_of_steps = 1; # arbitrary Initialisation
# if A_block_size[0] >= B_block_size[1] and length_of_matrices%A_block_size[0]==0:
#     no_of_steps = length_of_matrices/A_block_size[0]
# elif length_of_matrices%B_block_size[1] == 0:
#     no_of_steps = length_of_matrices/B_block_size[1]
# elif iProcs >= jProcs and length_of_matrices%iProcs == 0:
#     no_of_steps = length_of_matrices/iProcs
# elif length_of_matrices%jProcs == 0:
#     no_of_steps = length_of_matrices/jProcs
# elif length_of_matrices%nProcs == 0:
#     no_of_steps = length_of_matrices/nProcs # no physical reason behind this?
# else:
#     no_of_steps = length_of_matrices # mostly worst case. block size is 1

if rank == 0 and nProcs == 1:
    print("No. of processors = 1. Directly multiplying.")
    # for i in range(C_size[0]):
    #     for j in range(C_size[1]):
    #         for k in range(length_of_matrices):
    #             C[i][j] = C[i][j] + A[i][k]*B[k][j]
    C = A.dot(B)
    print("c_calc",C)
    print("c_act",C_act)
    print(np.amax(np.abs(C-C_act)))
    exit(0)

if rank == 0:
    print("No. of processors > 1. Proceeding to MPI execution.")

if length_of_matrices%iProcs != 0:
    print("Fox Algorithm not applicable!")
    exit(1)

B_block_size[0] = length_of_matrices/iProcs
A_block_size[1] = B_block_size[0]

no_of_steps = length_of_matrices/A_block_size[1]

if rank==0:
    print("Block sizes for A,B,C matrices are: ")
    print A_block_size
    print B_block_size
    print C_block_size

# basically, the time will scale linearly with the length of the matrix at this stage...

# Sending Array A to the processors
# First initialise a local A row for each processor

A_local_row = np.zeros((A_block_size[0],length_of_matrices),dtype='d')
if rank == 0:
    print("Communicating A to all processors.")

if (rank == 0):
    for proc_id in range(nProcs):
        rank_tag = 100 + proc_id
        rowblock_to_send = proc_id//jProcs
        i_start_index = rowblock_to_send*A_block_size[0]
        i_end_index = C_size[0]
        if(rowblock_to_send != iProcs-1):
            i_end_index = i_start_index+A_block_size[0]
        i_lim = np.arange(i_start_index,i_end_index,dtype='i')
        j_lim = np.arange(length_of_matrices,dtype='i')
        if proc_id == 0:
            A_local_row = A[np.ix_(i_lim,j_lim)]
        else:
            world.Send([A[np.ix_(i_lim,j_lim)],MPI.DOUBLE],dest=proc_id, tag=rank_tag)

if (rank != 0):
    world.Recv([A_local_row,MPI.DOUBLE],source=0,tag=(100+rank))

# Sending Array B to the processors
# First initialise a local block B for each processor

B_local_block = np.zeros((B_block_size[0],B_block_size[1]),dtype='d')
if rank == 0:
    print("Communicating B to all processors.")
if (rank == 0):
    for proc_id in range(nProcs):
        rank_tag = 200 + proc_id
        rowblock_to_send = proc_id//jProcs
        colblock_to_send = proc_id%jProcs
        i_start_index = rowblock_to_send*B_block_size[0]
        i_end_index = length_of_matrices
        if(rowblock_to_send != iProcs-1):
            i_end_index = i_start_index+B_block_size[0]
        j_start_index = colblock_to_send*B_block_size[1]
        j_end_index = C_size[1]
        if(colblock_to_send != jProcs-1):
            j_end_index = j_start_index+B_block_size[1]
        j_lim = np.arange(j_start_index,j_end_index,dtype='i')
        i_lim = np.arange(i_start_index,i_end_index,dtype='i')
        if proc_id == 0:
            B_local_block = B[np.ix_(i_lim,j_lim)]
        else:
            # print(np.size(B[np.ix_(i_lim,j_lim)]))
            world.Send([B[np.ix_(i_lim,j_lim)],MPI.DOUBLE],dest=proc_id, tag=rank_tag)

if (rank != 0):
    world.Recv([B_local_block,MPI.DOUBLE],source=0,tag=(200+rank))

# We don't have to do anything with C because we're going to write to only a particular value of C in each processor
C_local_block = np.zeros((C_block_size[0],C_block_size[1]),dtype='d')

# We have to initialise local A block

def refresh_A_block(rank,jProcs,A_block_size,A_local_row):
    A_block_number = rank//jProcs
    i_lim = np.arange(A_block_size[0],dtype='i')
    j_start_index = A_block_number*A_block_size[1]
    j_end_index = j_start_index+A_block_size[1]
    j_lim = np.arange(j_start_index,j_end_index,dtype='i')
    A_local_block = A_local_row[np.ix_(i_lim,j_lim)]
    return A_local_block

A_local_block=refresh_A_block(rank,jProcs,A_block_size,A_local_row);

# Now the local blocks are initialised. We need to proceed to the stepping part of the algorithm

def advance_B_blocks(rank,jProcs,nProcs,B_block_size,current_block):
    next_proc = (rank+jProcs) % nProcs
    prev_proc = (rank-jProcs) % nProcs
    send_rank_tag = 400 + next_proc
    receive_rank_tag = 400 + rank
    new_block = np.zeros((B_block_size[0],B_block_size[1]),dtype='d')
    print("Sending B block to",next_proc,"by processor",rank,"with tag",send_rank_tag)
    print("Receiving B block from",prev_proc,"by processor",rank,"with tag",receive_rank_tag)
    print(np.size(current_block))
    world.Send([current_block,MPI.DOUBLE],dest=next_proc,tag=send_rank_tag)
    world.Recv([new_block,MPI.DOUBLE],source=prev_proc,tag=receive_rank_tag)
    return new_block

print("Calculation started in Processor",rank)
for step_number in range(no_of_steps):
    C_local_block = C_local_block + A_local_block.dot(B_local_block)
    print("Step",step_number+1,"completed in processor",rank)
    A_local_row=np.roll(A_local_row,-1*int(A_block_size[1]),axis=1)
    A_local_block = refresh_A_block(rank,jProcs,A_block_size,A_local_row);
    print("A refreshed in step",step_number+1,"by processor",rank)
    B_local_block = advance_B_blocks(rank,jProcs,nProcs,B_block_size,B_local_block);
    print("B refreshed in step",step_number+1,"by processor",rank)

# print(C_local_block)
if rank == 0:
    print("Collecting the different blocks of C")
if (rank != 0):
    rank_tag = 300 + rank
    world.Send([C_local_block,MPI.DOUBLE],dest=0,tag=rank_tag)
else:
    i_lim = np.arange(0,C_block_size[0])
    j_lim = np.arange(0,C_block_size[1])
    C[np.ix_(i_lim,j_lim)] = C_local_block
    for proc in range(1,nProcs):
        # I,J is the left starting point of the big matrix C
        rank_tag = 300 + proc
        current_block = np.zeros((C_block_size[0],C_block_size[1]),dtype='d');
        world.Recv([current_block,MPI.DOUBLE],source=proc,tag=rank_tag)
        I = (proc//jProcs)*C_block_size[0]
        J = (proc%jProcs)*C_block_size[1]
        i_lim = np.arange(I,I+C_block_size[0])
        j_lim = np.arange(J,J+C_block_size[1])
        C[np.ix_(i_lim,j_lim)] = current_block
    print("c_calc",C)

if (rank == 0):
    print("c_act",C_act)
    print(np.amax(np.abs(C-C_act)))

print("Done",rank)
