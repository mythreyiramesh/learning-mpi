from mpi4py import MPI
import numpy as np
from temp1d_funcs import *
import matplotlib.pyplot as plt

world = MPI.COMM_WORLD
rank = world.Get_rank()
nProcs = world.Get_size()

global_grid_size = None; # arbitrary initialisation

if rank == 0:
    T = T_init(10000)
    col = 1
    clear_output_file()
    col = write_to_file_1d(T,col)
    global_grid_size = np.shape(T)[0];
    if global_grid_size%nProcs != 0:
        print("Change procs or grid size")
        exit(1)

global_grid_size = world.bcast(global_grid_size,root=0)
# print(grid_size,rank)

local_grid_size = global_grid_size/nProcs

# we are decomposing the domain to nProcs
T_local_prev = np.zeros(local_grid_size,dtype='d');
T_local_curr = np.zeros(local_grid_size,dtype='d');
# T_local_next = np.zeros(local_grid_size); # we don't need this, we are only using m for m+1
boundaries = np.array([0,0],dtype='d')

if rank == 0:
    for proc_id in range(nProcs):
        rank_tag = 100 + proc_id
        start_index = (proc_id)*local_grid_size
        end_index = start_index+local_grid_size
        if proc_id == 0:
            T_local_prev = T[start_index:end_index]
        else:
            world.Send([T[start_index:end_index],MPI.DOUBLE],dest=proc_id,tag=rank_tag)
else:
    world.Recv(T_local_prev,source=0,tag=(100+rank))

def get_boundary_vals(rank,nProcs,T_local_prev):
    bound_left = np.array([0],dtype='d')
    bound_right = np.array([0],dtype='d')
    # FIRST ATTEMPT
    # if rank == 0:
    #     bounds[0] = T_local_prev[0]
    #     next_proc =
    # elif rank == nProcs - 1:
    #     bounds[1] = T_local_prev[-1]
    # MAKE IT PBC
    # left_bounds = 200's, right_bounds = 300's
    left_proc = (rank-1)%nProcs
    right_proc = (rank+1)%nProcs
    left_recv_tag = 200 + rank
    right_recv_tag = 300 + rank
    left_send_tag = 300 + left_proc
    right_send_tag = 200 + right_proc

    # print("At rank",rank,"sending left value to",left_proc,"with tag",left_send_tag)
    world.Send([T_local_prev[0],MPI.DOUBLE],dest=left_proc,tag=left_send_tag)
    world.Send([T_local_prev[-1],MPI.DOUBLE],dest=right_proc,tag=right_send_tag)
    # print("At rank",rank,"receiving from",left_proc,"with tag",left_recv_tag)
    world.Recv(bound_left,source=left_proc,tag=left_recv_tag)
    world.Recv(bound_right,source=right_proc,tag=right_recv_tag)

    return np.array([bound_left,bound_right],dtype='d')

# if rank == 0:
#     print(T_local_curr)
# plt.plot(T_local_prev)

total_steps = 100;
# first let it perform calculations for all the inside nodes, this is independent of the processor.
for step in range(total_steps):
    boundaries = get_boundary_vals(rank,nProcs,T_local_prev)
    T_local_curr[0] = 0.5*(boundaries[0]+T_local_prev[1])
    for i in range(1,local_grid_size-1):
        T_local_curr[i] = 0.5*(T_local_prev[i-1] + T_local_prev[i+1])
    T_local_curr[-1] = 0.5*(boundaries[1]+T_local_prev[-2])
    T_local_prev = T_local_curr
    if step%10 == 0:
        if rank != 0:
            world.Send([T_local_prev,MPI.DOUBLE],dest=0,tag=step+rank)
        else:
            T_curr = np.zeros(global_grid_size,dtype='d')
            for proc_id in range(nProcs):
                start_index = (proc_id)*local_grid_size
                end_index = start_index+local_grid_size
                if proc_id == 0:
                    T_curr[start_index:end_index] = T_local_curr
                else:
                    world.Recv(T_curr[start_index:end_index],source=proc_id,tag=step+proc_id)
            col = write_to_file_1d(T_curr,col)
    # if rank == 0:
    #     print T_local_prev

# print(T_local_curr)

# plt.plot(T_local_curr)
# plt.show()
print("Done",rank)
