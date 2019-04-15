from mpi4py import MPI
import numpy as np
from temp1d_basic_funcs import *
import time as t

world = MPI.COMM_WORLD
rank = world.Get_rank()
nProcs = world.Get_size()

alpha = 1.0
delT = 0.5
delX = 1.0
prefactor = (alpha*delT)/(delX**2)

global_grid_size = np.array([0],dtype='i'); # arbitrary initialisation
local_grid_size = np.array([0],dtype='i'); # arbitrary initialisation
tolerance = 10**(-4) # tolerance
# print "Set Tolerance:",tolerance

global_error = np.array([10**5],dtype='d'); # arbitrary large number
local_error = np.array([10**5],dtype='d'); # arbitrary large number

if rank == 0:
    domain_size = 200
    # print "Initialising the domain on processor",rank
    T = T_1Dinit(domain_size)
    BC_left = np.zeros(1,dtype='d');
    BC_right = np.zeros(1,dtype='d');
    time_taken = 0;
    # print "Boundary conditions set at processor",rank
    layer = 1
    clear_output_file()
    layer = write_to_file_1d(T,layer)
    global_grid_size = np.shape(T)[0];
    local_grid_size = np.array([int(global_grid_size/nProcs)],dtype='i')
    # print "Communicating local grid sizes to each processor"
    for proc_id in range(nProcs):
        to_send = np.copy(local_grid_size)
        if proc_id != nProcs-1:
            world.Send([to_send,MPI.INT],dest=proc_id,tag=(100+proc_id))
        else:
            if global_grid_size%nProcs == 0:
                to_send[0] = local_grid_size[0]
            else:
                to_send[0] = global_grid_size-(nProcs-1)*local_grid_size[0]
            world.Send([to_send,MPI.INT],dest=proc_id,tag=(100+proc_id))
else:
    world.Recv(local_grid_size,source=0,tag=(100+rank))

global_grid_size = world.bcast(global_grid_size,root=0)

# print "Initialising local grids for computation"
T_local_prev = np.zeros(local_grid_size+2,dtype='d');
T_local_curr = np.zeros(local_grid_size+2,dtype='d');

# This whole part can be eliminated if we're going to initialise randomly on each processor
if rank == 0:
    for proc_id in range(nProcs):
        rank_tag = 200 + proc_id
        col_start_index = proc_id*local_grid_size[0]
        if proc_id != nProcs-1:
            col_end_index = col_start_index+local_grid_size[0]
        else:
            col_end_index = global_grid_size
        if proc_id == 0:
            T_local_prev[1:-1] = T[col_start_index:col_end_index]
        else:
            world.Send([T[col_start_index:col_end_index],MPI.DOUBLE],dest=proc_id,tag=rank_tag)
else:
    recv_array = np.zeros(local_grid_size,dtype='d')
    world.Recv(recv_array,source=0,tag=(200+rank))
    T_local_prev[1:-1] = recv_array

# print "Initialisation complete on processor",rank

def communicate_horizontal_BC_from_master(rank,nProcs,local_grid_size):
    # Special channel to communicate right boundary values: 450
    if rank == 0:
        global BC_right
        world.Send([BC_right,MPI.DOUBLE],dest=nProcs-1,tag=450+nProcs-1)
    else:
        if rank == nProcs-1:
            BC_right = np.zeros((global_grid_size,1),dtype='d')
            world.Recv(BC_right,source=0,tag=450+rank)
    return

def get_horizontal_boundary_vals(rank,T_local_prev):
    # left_bounds = 300's, right_bounds = 400's
    left_proc = (rank-1)%nProcs
    right_proc = (rank+1)%nProcs
    if rank != 0:
        left_recv_tag = 300 + rank
        left_send_tag = 400 + left_proc
    if rank != nProcs-1:
        right_recv_tag = 400 + rank
        right_send_tag = 300 + right_proc
    if rank != 0:
        world.Send([np.ascontiguousarray(T_local_prev[1],dtype='d'),MPI.DOUBLE],dest=left_proc,tag=left_send_tag)
    if rank != nProcs-1:
        world.Send([np.ascontiguousarray(T_local_prev[-2],dtype='d'),MPI.DOUBLE],dest=right_proc,tag=right_send_tag)
    if rank != 0:
        recv_array = np.zeros(1,dtype='d')
        world.Recv(recv_array,source=left_proc,tag=left_recv_tag)
        T_local_prev[0] = recv_array
    else:
        col_start_index = rank*(int(global_grid_size)/nProcs)
        col_end_index = col_start_index+local_grid_size[0]
        T_local_prev[0] = BC_left[0]
    if rank != nProcs-1:
        recv_array = np.zeros(1,dtype='d')
        world.Recv(recv_array,source=right_proc,tag=right_recv_tag)
        T_local_prev[-1] = recv_array
    else:
        col_start_index = rank*(int(global_grid_size/nProcs))
        col_end_index = col_start_index+local_grid_size[0]
        T_local_prev[-1] = BC_right[0]
    return T_local_prev

step = 1
# print "Communicating external boundary conditions to boundary processors"
communicate_horizontal_BC_from_master(rank,nProcs,local_grid_size)

# print "Computation starts on processor",rank
while (global_error[0]>tolerance):
    if rank == 0:
        t1 = t.time()
    T_local_prev = get_horizontal_boundary_vals(rank,T_local_prev)
    T_local_curr[1:-1] = T_local_prev[1:-1]+prefactor*(T_local_prev[:-2]-2*T_local_prev[1:-1]+T_local_prev[2:])
    local_error[0] = np.sum(np.power(np.abs(T_local_curr[1:-1]-T_local_prev[1:-1]),2))
    T_local_prev = np.copy(T_local_curr)
    if rank == 0:
        t2 = t.time()
        time_taken = time_taken + t2 - t1
    world.Allreduce(local_error,global_error,op=MPI.MAX)
    step = step + 1
else:
    print "Global tolerance reached"
    if rank != 0:
        world.Send([T_local_prev[1:-1],MPI.DOUBLE],dest=0,tag=(100*step)+rank)
        # print(np.shape(T_local_prev[1:-1]),rank,"sent")
    else:
        T_curr = np.zeros(global_grid_size,dtype='d')
        for proc_id in range(nProcs):
            rank_tag = (100*step)+proc_id
            col_start_index = proc_id*local_grid_size[0]
            # col_end_index = col_start_index+local_grid_size[0]
            if proc_id != nProcs-1:
                col_end_index = col_start_index+local_grid_size[0]
            else:
                col_end_index = global_grid_size
            # print(col_start_index,proc_id,col_end_index)
            if proc_id == 0:
                T_curr[col_start_index:col_end_index] = T_local_prev[1:-1]
            else:
                recv_array = np.zeros((col_end_index-col_start_index),dtype='d')
                # print(np.shape(recv_array),proc_id,"recv")
                world.Recv(recv_array,source=proc_id,tag=(100*step)+proc_id)
                T_curr[col_start_index:col_end_index] = recv_array
    if rank == 0:
        layer = write_to_file_1d(T_curr,layer)

if rank == 0:
    print("time taken:",time_taken)
    print("steps:",step)
    print("global error",global_error[0])
