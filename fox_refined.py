# Fox Algorithm to multiply two matrices using MPI

# Note: Initialisation can be done by one processor and later the same processor can start doing work. Another option (if you're reading from a file, use different processors to read from different parts of the file)

from mpi4py import MPI
import numpy as np
from math import floor
from funcs_refined import *

world = MPI.COMM_WORLD
rank = world.Get_rank()
nprocs = world.Get_size()
C_size = np.array([1,1],dtype='i'); # something arbitrary
C_block_size = np.array([0, 0],dtype='i') # something arbitrary
length_of_matrices = np.array([1],dtype='i'); # something arbitrary

if (rank == 0):
    # upper cap = 100, lower cap = -100
    [A,B] = init_input_matrices([4,4],[4,16],-10,10)
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
    for i in range(1,nprocs):
        world.Send([C_size,MPI.INT],dest=i,tag=1)
        world.Send([length_of_matrices,MPI.INT],dest=i,tag=2)
else:
    world.Recv([C_size,MPI.INT],source=0)
    world.Recv([length_of_matrices,MPI.INT],source=0)

[iProcs,jProcs] = get_procs(C_size,nprocs);
print("iProcs jProcs",iProcs,jProcs)
length_of_matrices = int(length_of_matrices)
C_block_size[0] = C_size[0]/iProcs
C_block_size[1] = C_size[1]/jProcs
A_block_size[0] = C_block_size[0]
A_block_size[1] = length_of_matrices/jProcs
B_block_size[0] = length_of_matrices/iProcs
B_block_size[1] = C_block_size[1]

print("BlockSize",C_block_size)

no_of_steps = length_of_matrices/C_block_size[1] # there will be an issue here if the blocks are not square

# Sending Array A to the processors
# First initialise a local A row for each processor
A_local_row = np.zeros((C_block_size[0],length_of_matrices),dtype='d')

# There will be same number of blocks in a dimension as the number of processors for that dimension. Hence, the block length will determine the start and end indices of the blocks along j.
if (rank == 0):
    # print(A)
    for i in range(0,jProcs):
        for j in range(0,iProcs):
            proc_id = i*(iProcs)+j
            rank_tag = 100 + proc_id # this is arbitrary
            start_index = (i)*C_block_size[0]
            end_index = C_size[0]
            if (i != jProcs-1):
                end_index = (i+1)*C_block_size[0]

            if proc_id == 0:
                A_local_row = A[start_index:end_index][:]
            else:
                world.Send([A[start_index:end_index][:],MPI.DOUBLE],dest=proc_id, tag=rank_tag)

if (rank != 0):
    world.Recv([A_local_row,MPI.DOUBLE],source=0,tag=(100+rank))

# Sending Array B to the processors
# First initialise a local block B for each processor
B_local_block = np.zeros((C_block_size[0],C_block_size[1]),dtype='d')

if (rank == 0):
    for i in range(0,jProcs):
        for j in range(0,iProcs):
            proc_id = i*(iProcs)+j
            rank_tag = 200 + proc_id # this is arbitrary
            i_start_index = (proc_id//jProcs)*C_block_size[0]
            j_start_index = (proc_id%jProcs)*C_block_size[1]
            i_lim = np.arange(i_start_index,i_start_index+C_block_size[1])
            j_lim = np.arange(j_start_index,j_start_index+C_block_size[0])
            if proc_id == 0:
                B_local_block = B[np.ix_(i_lim,j_lim)]
            else:
                world.Send([B[np.ix_(i_lim,j_lim)],MPI.DOUBLE],dest=proc_id, tag=rank_tag)

if (rank != 0):
    world.Recv([B_local_block,MPI.DOUBLE],source=0,tag=(200+rank))

# We don't have to do anything with C because we're going to write to only a particular value of C in each processor
C_local_block = np.zeros((C_block_size[0],C_block_size[1]),dtype='d')

# We have to initialise local A block
# (MUST be modularised)
# We can find start index by getting the quotient of the processor and iProcs
A_row_start_index = (rank // iProcs)*C_block_size[0]
A_row_end_index = A_row_start_index+C_block_size[1]
i_lim = np.arange(0,C_block_size[1])
j_lim = np.arange(A_row_start_index,A_row_end_index)
A_local_block = A_local_row[np.ix_(i_lim,j_lim)]

# Now the local blocks are initialised. We need to proceed to the stepping part of the algorithm

# note: there are three rings here, hence, must be careful to not let the communication remain stalled!
def advance_B_blocks(current_block,C_block_size,nproc):
    next_proc = (rank+jProcs) % nproc
    prev_proc = (rank-jProcs) % nproc
    new_block = np.zeros((C_block_size[0],C_block_size[1]),dtype='d')
    world.Send([current_block,MPI.DOUBLE],dest=next_proc,tag=3)
    world.Recv([new_block,MPI.DOUBLE],source=prev_proc,tag=3)
    return new_block

for step_number in range(no_of_steps):
    C_local_block = C_local_block + A_local_block.dot(B_local_block)
    A_row_start_index = (A_row_start_index + C_block_size[1]) % length_of_matrices
    A_row_end_index = (A_row_start_index + C_block_size[1]) % length_of_matrices
    if A_row_end_index == 0:
        A_row_end_index = length_of_matrices
    a_dim = np.arange(0,C_block_size[0]);
    b_dim = np.arange(A_row_start_index,A_row_end_index) ;
    A_local_block = A_local_row[np.ix_(a_dim,b_dim)]
    B_local_block = advance_B_blocks(B_local_block,C_block_size,nprocs);


if (rank != 0):
    rank_tag = 300 + rank
    world.Send([C_local_block,MPI.DOUBLE],dest=0,tag=rank_tag)
else:
    i_lim = np.arange(0,C_block_size[1])
    j_lim = np.arange(0,C_block_size[0])
    C[np.ix_(i_lim,j_lim)] = C_local_block
    for proc in range(1,nprocs):
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

# print("Done",rank)
