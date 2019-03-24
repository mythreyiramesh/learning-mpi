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
nProcs = world.Get_size() # for now, this will be 4
C_size = np.array([1,1],dtype='i'); # something arbitrary
A_block_size = np.array([0, 0],dtype='i') # something arbitrary
B_block_size = np.array([0, 0],dtype='i') # something arbitrary
C_block_size = np.array([0, 0],dtype='i') # something arbitrary
length_of_matrices = np.array([1],dtype='i'); # something arbitrary

if (rank == 0):
    # upper cap = 100, lower cap = -100
    [A,B] = init_input_matrices([4,4],[4,8],-10,10)
    A_size = np.shape(A);
    B_size = np.shape(B);
    if A_size[1] != B_size[0]:
        print("Matrices cannot be multiplied")
        exit(1)
    else:
        length_of_matrices[0] = A_size[1]
    C_act = A.dot(B)
    # print("Actual Ans:")
    # print(C_act)
    C_size = np.array([A_size[0],B_size[1]],dtype='i')
    # print(type(length_of_matrices))
    C = np.zeros(C_size,dtype='d')
    for i in range(1,nProcs):
        world.Send([C_size,MPI.INT],dest=i,tag=1)
        world.Send([length_of_matrices,MPI.INT],dest=i,tag=2)
else:
    world.Recv([C_size,MPI.INT],source=0)
    world.Recv([length_of_matrices,MPI.INT],source=0)

[iProcs,jProcs] = get_procs(C_size,nProcs);
length_of_matrices = int(length_of_matrices)

# print(iProcs,jProcs)
C_block_size[0] = C_size[0]/iProcs
C_block_size[1] = C_size[1]/jProcs

# It might make more sense to keep these separate... Because they are of different shapes if the matrix is rectangular at the end and the number of processors isn't enough to factorise according to the aspect ratio of the matrices. Can extend to rough matrices later... (where the last block needs a separate treatment)

A_block_size[0] = C_block_size[0]
B_block_size[1] = C_block_size[1]

# By rules of matrix product, A_block_size[1] must always equal B_block_size[0]. This value can be anything! Probably must be optimised based on number of processors, length_of_matrices etc... For now, pick it to be such that at least one of the blocks are going to be square. But this is arbitrary.

no_of_steps = 1; # arbitrary Initialisation
if A_block_size[0] <= B_block_size[1] and length_of_matrices%A_block_size[0]==0:
    no_of_steps = length_of_matrices/A_block_size[0]
elif length_of_matrices%B_block_size[1] == 0:
    no_of_steps = length_of_matrices/B_block_size[1]
elif iProcs <= jProcs and length_of_matrices%iProcs == 0:
    no_of_steps = length_of_matrices/iProcs
elif length_of_matrices%jProcs == 0:
    no_of_steps = length_of_matrices/jProcs
elif length_of_matrices%nProcs == 0:
    no_of_steps = length_of_matrices/nProcs # no physical reason behind this?
else:
    no_of_steps = length_of_matrices # mostly worst case. block size is 1

A_block_size[1] = length_of_matrices/no_of_steps
B_block_size[0] = A_block_size[1]

# basically, the time will scale linearly with the length of the matrix at this stage...

# if rank == 0:
#     if length_of_matrices%jProcs or length_of_matrices%iProcs:
#         print("Matrices have rough edges, current implementation of the algorithm doesn't allow for this multiplication. Please try with other matrices.")
#         exit(1)

# print("BlockSize",C_block_size)

# print(C_block_size)
# no_of_steps = length_of_matrices/A_block_size[1]
# print(length_of_matrices,no_of_steps)

# doubt: are blocks always square?? Looks like it...

# note: since rank 0 is going to take charge of sending information to other processors, I don't have to send anything to rank 0 itself. this might change if the implementation changes

# Sending Array A to the processors
# First initialise a local A row for each processor
# if (rank != 0):
A_local_row = np.zeros((A_block_size[0],length_of_matrices),dtype='d')

# this can be combined into the other if condition, but keeping it here for clarity

# There will be same number of blocks in i dimension as the number of processors for that dimension. Hence, the block length will determine the start and end indices of the blocks along j.
if (rank == 0):
    # print(A)
    # # OLD
    # for i in range(0,jProcs): # is it iProcs or jProcs?
    #     for j in range(0,iProcs):
    #         proc_id = i*(iProcs)+j
    #         # print("Sending to",proc_id)
    #         rank_tag = 100 + proc_id # this is arbitrary
    #         start_index = (i)*A_block_size[0]
    #         end_index = C_size[0]
    #         if (i != jProcs-1):
    #             end_index = (i+1)*A_block_size[0]
    #
    #         if proc_id == 0:
    #             A_local_row = A[start_index:end_index][:]
    #         else:
    #             world.Send([A[start_index:end_index][:],MPI.DOUBLE],dest=proc_id, tag=rank_tag)
    #         # print('Sent A',start_index,'to',end_index,'towards rank',proc_id)
    # NEW
    for proc_id in range(nProcs):
        rank_tag = 100 + proc_id
        # rowblock_to_send = int(rank/jProcs);
        rowblock_to_send = rank//jProcs
        start_index = rowblock_to_send*A_block_size[0]
        end_index = C_size[0]
        if(proc_id != nProcs-1):
            end_index = start_index+A_block_size[0]
        if proc_id == 0:
            A_local_row = A[start_index:end_index][:]
        else:
            world.Send([A[start_index:end_index][:],MPI.DOUBLE],dest=proc_id, tag=rank_tag)

if (rank != 0):
    world.Recv([A_local_row,MPI.DOUBLE],source=0,tag=(100+rank))
    # print(rank,'received A',A_local_row,'from',0)


# Sending Array B to the processors
# First initialise a local block B for each processor
# if (rank != 0):
B_local_block = np.zeros((C_block_size[0],C_block_size[1]),dtype='d')
# print(np.shape(B_local_block),rank)
# this can be combined into the other if condition, but keeping it here for clarity

if (rank == 0):
    # print(B)
    # print(np.shape(B))
    # # OLD
    # for i in range(0,jProcs): # is it iProcs or jProcs?
    #     for j in range(0,iProcs):
    #         proc_id = i*(iProcs)+j
    #         # print("Sending to",proc_id)
    #         rank_tag = 200 + proc_id # this is arbitrary
    #         # doubt : since the start index and end index along the j dimension (therefore i) is always fixed as 0 and C_block_size[j], maybe it's just best to initialise as that or send B only from 0 to C_block_size[0]. NO NO BIG NO
    #         # # OLD
    #         # i_start_index = (i)*C_block_size[0]
    #         # i_end_index = C_size[0]
    #         # j_start_index = (j)*C_block_size[1]
    #         # j_end_index = C_size[1]
    #         # if (i != iProcs-1):
    #         #     i_end_index = (i+1)*C_block_size[0]
    #         # if (j != jProcs-1):
    #         #     j_end_index = (j+1)*C_block_size[1]
    #         # NEW
    #         i_start_index = (proc_id//jProcs)*C_block_size[0]
    #         j_start_index = (proc_id%jProcs)*C_block_size[1]
    #         # print i_start_index
    #         # print j_start_index
    #         # print i_start_index+C_block_size[1]
    #         # print j_start_index+C_block_size[0]
    #         i_lim = np.arange(i_start_index,i_start_index+C_block_size[1])
    #         j_lim = np.arange(j_start_index,j_start_index+C_block_size[0])
    #         # print B[np.ix_([i_start_index,i_end_index-1],[j_start_index,j_end_index-1])]
    #         # print B[np.ix_(i_lim,j_lim)]
    #         if proc_id == 0:
    #             B_local_block = B[np.ix_(i_lim,j_lim)]
    #             # print([i_start_index,i_end_index+1],[j_start_index,j_end_index+1])
    #             # print B[np.ix_([i_start_index,i_end_index-1],[j_start_index,j_end_index-1])]
    #             # print(np.shape(B_local_block))
    #         else:
    #             world.Send([B[np.ix_(i_lim,j_lim)],MPI.DOUBLE],dest=proc_id, tag=rank_tag)
    #             # print('Sent B',i_start_index,'to',i_end_index,'and',j_start_index,'to',j_end_index,'towards rank',proc_id)
    # NEW
    for proc_id in range(nProcs):
        rowblock_to_send = rank//jProcs
        colblock_to_send = rank%jProcs
        i_start_index = rowblock_to_send*B_block_size[0]
        j_start_index = colblock_to_send*B_block_size[1]
        i_end_index = length_of_matrices
        if rowblock_to_send != length_of_matrices//B_block_size[0]:
            i_end_index = i_start_index+B_block_size[0]
        j_end_index = C_size[1]
        if colblock_to_send != C_size[1]//B_block_size[1]:
            j_end_index = j_start_index+B_block_size[1]
        i_lim = np.arange(i_start_index,i_end_index,dtype='i')
        j_lim = np.arange(j_start_index,j_end_index,dtype='i')
        if proc_id == 0:
            B_local_block = B[np.ix_(i_lim,j_lim)]
        else:
            world.Send([B[np.ix_(i_lim,j_lim)],MPI.DOUBLE],dest=proc_id, tag=rank_tag)

if (rank != 0):
    world.Recv([B_local_block,MPI.DOUBLE],source=0,tag=(200+rank))
    # print(rank,'received B',B_local_block,'from',0)

# We don't have to do anything with C because we're going to write to only a particular value of C in each processor
C_local_block = np.zeros((C_block_size[0],C_block_size[1]),dtype='d')

# We have to initialise local A block

# # OLD
# # (MUST be modularised)
# # We can find start index by getting the quotient of the processor and iProcs
# # print A_local_row, rank
# A_row_start_index = (rank // iProcs)*C_block_size[0]
# A_row_end_index = A_row_start_index+C_block_size[1]
# # print A_row_start_index, A_row_end_index, rank
# i_lim = np.arange(0,C_block_size[1])
# j_lim = np.arange(A_row_start_index,A_row_end_index)
# A_local_block = A_local_row[np.ix_(i_lim,j_lim)]
# # print A_local_block,rank
# NEW
A_block_number = rank//jProcs
i_lim = np.arange(A_block_size[0],dtype='i')
j_start_index = A_block_number*A_block_size[1]
j_end_index = j_start_index+A_block_size[1]
j_lim = np.arange(j_start_index,j_end_index,dtype='i')
A_local_block = A_local_row[np.ix_(i_lim,j_lim)]
# Note to self: use an object to store the start and end indices
# print(np.shape(A_local_block),np.shape(B_local_block),rank)

# Now the local blocks are initialised. We need to proceed to the stepping part of the algorithm

# note: there are three rings here, hence, must be careful to not let the communication remain stalled! Must check!!
#####
def advance_B_blocks(current_block,C_block_size,nproc):
    # note to self: use global C_block_size??
    next_proc = (rank+jProcs) % nproc
    prev_proc = (rank-jProcs) % nproc
    # print "prev",prev_proc,"curr",rank,"next",next_proc

    new_block = np.zeros((C_block_size[0],C_block_size[1]),dtype='d')
    # if (rank%jProcs !=0):
    #     world.Recv([new_block,MPI.DOUBLE],source=prev_proc,tag=3)
    #     print(rank,'received from',prev_proc)
    #     world.Send([current_block,MPI.DOUBLE],dest=next_proc,tag=3)
    #     print(rank,'sent to',next_proc)
    # else:
    #     world.Send([current_block,MPI.DOUBLE],dest=next_proc,tag=3)
    #     print(rank,'sent to',next_proc)
    #     world.Recv([new_block,MPI.DOUBLE],source=prev_proc,tag=3)
    #     print(rank,'received from',prev_proc)
    world.Send([current_block,MPI.DOUBLE],dest=next_proc,tag=3)
    # print(rank,'sent to',next_proc)
    world.Recv([new_block,MPI.DOUBLE],source=prev_proc,tag=3)
    # print(rank,'received from',prev_proc)
    return new_block

for step_number in range(no_of_steps):
    # print(np.shape(A_local_block),np.shape(B_local_block),rank)
    C_local_block = C_local_block + A_local_block.dot(B_local_block)
    # advance_A_blocks();
    # Use roll() to make it better!
    A_row_start_index = (A_row_start_index + C_block_size[1]) % length_of_matrices
    A_row_end_index = (A_row_start_index + C_block_size[1]) % length_of_matrices
    # print A_row_end_index
    if A_row_end_index == 0:
        A_row_end_index = length_of_matrices
    a_dim = np.arange(0,C_block_size[0]);
    b_dim = np.arange(A_row_start_index,A_row_end_index) ;
    # if b_dim.any() == 0:
    #     print(rank,step_number,A_row_start_index)
    # print a_dim,b_dim
    # A_local_block = A_local_row[np.ix_([0,C_block_size[0]-1],[A_row_start_index,A_row_end_index-1])]
    A_local_block = A_local_row[np.ix_(a_dim,b_dim)]
    B_local_block = advance_B_blocks(B_local_block,C_block_size,nProcs);

# print(C_local_block)


if (rank != 0):
    rank_tag = 300 + rank
    world.Send([C_local_block,MPI.DOUBLE],dest=0,tag=rank_tag)
else:
    i_lim = np.arange(0,C_block_size[1])
    j_lim = np.arange(0,C_block_size[0])
    C[np.ix_(i_lim,j_lim)] = C_local_block
    for proc in range(1,nProcs):
        # I,J is the left starting point of the big matrix C
        rank_tag = 300 + proc
        current_block = np.zeros((C_block_size[0],C_block_size[1]),dtype='d');
        world.Recv([current_block,MPI.DOUBLE],source=proc,tag=rank_tag)
        I = (proc//jProcs)*C_block_size[0]
        J = (proc%jProcs)*C_block_size[1]
        i_lim = np.arange(I,I+C_block_size[1])
        j_lim = np.arange(J,J+C_block_size[0])
        C[np.ix_(i_lim,j_lim)] = current_block
    print C

if (rank == 0):
    print(C_act)
    print(np.amax(np.abs(C-C_act)))
#####
print("Done",rank)
