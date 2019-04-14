from mpi4py import MPI
import numpy as np
from couette_funcs import *
import time as t
import warnings

warnings.filterwarnings('error')

world = MPI.COMM_WORLD
rank = world.Get_rank()
nProcs = world.Get_size()
iProcs = np.zeros(1,dtype='i')
jProcs = np.zeros(1,dtype='i')

rho = 1.0
mu = 1.0
delT = 0.5
delY = 1.0
prefactor = (mu*delT)/(rho*(delY)**2)

global_grid_size = np.array([0,0],dtype='i'); # arbitrary initialisation
local_grid_size = np.array([0,0],dtype='i'); # arbitrary initialisation
tolerance = 10**(-4) # tolerance
print "Set Tolerance:",tolerance

global_error = np.array([10**5],dtype='d'); # arbitrary large number
local_error = np.array([10**5],dtype='d'); # arbitrary large number

if rank == 0:
    domain_width = 100
    domain_height = 50
    print "Initialising the domain on processor",rank
    U = U_2Dinit([domain_height,domain_width])
    # BC_left = np.zeros((domain_height,1),dtype='d');
    # BC_right = np.zeros((domain_height,1),dtype='d');
    BC_top = np.zeros((1,domain_width),dtype='d');
    BC_bottom = np.zeros((1,domain_width),dtype='d');
    time_taken = 0;
    print "Boundary conditions set at processor",rank
    layer = 1
    clear_output_file()
    layer = write_to_file_2d(U,layer)
    global_grid_size = np.shape(U);
    iProcs,jProcs = factor_procs(nProcs,global_grid_size)
    local_grid_size = np.array([int(global_grid_size[0]/iProcs),int(global_grid_size[1]/jProcs)],dtype='i')
    print "Communicating local grid sizes to each processor"
    for proc_id in range(nProcs):
        i_part =  int(proc_id/jProcs)
        j_part = proc_id%jProcs
        to_send = np.copy(local_grid_size)
        if i_part != iProcs-1 and j_part != jProcs-1:
            world.Send([local_grid_size,MPI.INT],dest=proc_id,tag=(100+proc_id))
        else:
            if i_part == iProcs-1:
                if global_grid_size[0]%iProcs == 0:
                    to_send[0] = local_grid_size[0]
                else:
                    to_send[0] = global_grid_size[0]-(iProcs-1)*local_grid_size[0]
            if j_part == jProcs-1:
                if global_grid_size[0]%jProcs == 0:
                    to_send[1] = local_grid_size[1]
                else:
                    to_send[1] = global_grid_size[1]-(jProcs-1)*local_grid_size[1]
            world.Send([to_send,MPI.INT],dest=proc_id,tag=(100+proc_id))
else:
    world.Recv(local_grid_size,source=0,tag=(100+rank))

global_grid_size = world.bcast(global_grid_size,root=0)
iProcs = world.bcast(iProcs,root=0)
jProcs = world.bcast(jProcs,root=0)

print "Initialising local grids for computation"
U_local_old = np.zeros(local_grid_size+2,dtype='d');
U_local_new = np.zeros(local_grid_size+2,dtype='d');

# This whole part can be eliminated if we're going to initialise randomly on each processor
if rank == 0:
    for i_part in range(iProcs):
        for j_part in range(jProcs):
            proc_id = i_part*jProcs + j_part
            rank_tag = 200 + proc_id
            row_start_index = i_part*local_grid_size[0]
            col_start_index = j_part*local_grid_size[1]
            if i_part != iProcs-1:
                row_end_index = row_start_index+local_grid_size[0]
            else:
                row_end_index = global_grid_size[0]
            if j_part != jProcs-1:
                col_end_index = col_start_index+local_grid_size[1]
            else:
                col_end_index = global_grid_size[1]
            if proc_id == 0:
                U_local_old[1:-1,1:-1] = U[row_start_index:row_end_index,col_start_index:col_end_index]
            else:
                world.Send([np.ascontiguousarray(U[row_start_index:row_end_index,col_start_index:col_end_index],dtype='d'),MPI.DOUBLE],dest=proc_id,tag=rank_tag)
else:
    world.Recv(np.ascontiguousarray(U_local_old[1:-1,1:-1],dtype='d'),source=0,tag=(200+rank))

print "Initialisation complete on processor",rank

# def communicate_horizontal_BC_from_master(rank,nProcs,jProcs,local_grid_size):
#     # Special channel to communicate right boundary values: 450
#     # Special channel to communicate left boundary values: 350
#     if rank == 0:
#         global BC_right, BC_left
#         for proc_id in range(1,nProcs):
#             if proc_id%jProcs == jProcs-1:
#                 world.Send([BC_right,MPI.DOUBLE],dest=proc_id,tag=450+proc_id)
#             if proc_id%jProcs == 0:
#                 world.Send([BC_left,MPI.DOUBLE],dest=proc_id,tag=350+proc_id)
#     else:
#         if rank%jProcs == jProcs-1:
#             BC_right = np.zeros((global_grid_size[0],1),dtype='d')
#             world.Recv(BC_right,source=0,tag=450+rank)
#         if rank%jProcs == 0:
#             BC_left = np.zeros((global_grid_size[0],1),dtype='d')
#             world.Recv(BC_left,source=0,tag=350+rank)
#     return

def communicate_vertical_BC_from_master(rank,nProcs,jProcs,local_grid_size):
    # Special channel to communicate top boundary values: 550
    # Special channel to communicate bottom boundary values: 650
    if rank == 0:
        global BC_bottom,BC_top
        for proc_id in range(1,nProcs):
            if int(proc_id/jProcs) == iProcs-1:
                world.Send([BC_bottom,MPI.DOUBLE],dest=proc_id,tag=650+proc_id)
            if int(proc_id/jProcs) == 0:
                world.Send([BC_top,MPI.DOUBLE],dest=proc_id,tag=550+proc_id)
    else:
        if int(rank/jProcs) == iProcs-1:
            BC_bottom = np.zeros((1,global_grid_size[1]),dtype='d')
            world.Recv(BC_bottom,source=0,tag=650+rank)
        if int(rank/jProcs) == 0:
            BC_top = np.zeros((1,global_grid_size[1]),dtype='d')
            world.Recv(BC_top,source=0,tag=550+rank)
    return

def get_horizontal_boundary_vals(rank,jProcs,U_local_old):
    # left_bounds = 300's, right_bounds = 400's
    i_part = int(rank/jProcs)
    j_part = rank%jProcs
    left_proc = (i_part)*jProcs+(j_part-1)%jProcs
    right_proc = (i_part)*jProcs+(j_part+1)%jProcs
    if j_part != 0:
        left_recv_tag = 300 + rank
        left_send_tag = 400 + left_proc
    if j_part != jProcs-1:
        right_recv_tag = 400 + rank
        right_send_tag = 300 + right_proc
    if j_part != 0:
        world.Send([np.ascontiguousarray(U_local_old[1:-1,1],dtype='d'),MPI.DOUBLE],dest=left_proc,tag=left_send_tag)
    if j_part != jProcs-1:
        world.Send([np.ascontiguousarray(U_local_old[1:-1,-1],dtype='d'),MPI.DOUBLE],dest=right_proc,tag=right_send_tag)
    if j_part != 0:
        world.Recv(np.ascontiguousarray(U_local_old[1:-1,0],dtype='d'),source=left_proc,tag=left_recv_tag)
    else:
        row_start_index = (i_part)*(int(global_grid_size[0])/iProcs)
        row_end_index = row_start_index+local_grid_size[0]
        U_local_old[1:-1,0] = U_local_old[1:-1,1]
        # print(np.amin(U_local_old[1:-1,1]),np.amax(U_local_old[1:-1,1]))
    if j_part != jProcs-1:
        world.Recv(np.ascontiguousarray(U_local_old[1:-1,-1],dtype='d'),source=right_proc,tag=right_recv_tag)
    else:
        row_start_index = (i_part)*(int(global_grid_size[0]/iProcs))
        row_end_index = row_start_index+local_grid_size[0]
        U_local_old[1:-1,-1] = U_local_old[1:-1,-2]
        # print(np.amin(U_local_old[1:-1,-2]),np.amax(U_local_old[1:-1,-2]))
    return

def get_vertical_boundary_vals(rank,jProcs,iProcs,U_local_old):
    # top_bounds = 500's, bottom_bounds = 600's
    i_part = int(rank/jProcs)
    j_part = rank%jProcs
    top_proc = ((i_part-1)%iProcs)*jProcs+ j_part
    bottom_proc = ((i_part+1)%iProcs)*jProcs+ j_part
    if i_part != 0:
        top_recv_tag = 500 + rank
        top_send_tag = 600 + top_proc
    if i_part != iProcs-1:
        bottom_recv_tag = 600 + rank
        bottom_send_tag = 500 + bottom_proc
    if i_part != 0:
        world.Send([np.ascontiguousarray(U_local_old[1,1:-1],dtype='d'),MPI.DOUBLE],dest=top_proc,tag=top_send_tag)
    if i_part != iProcs-1:
        world.Send([np.ascontiguousarray(U_local_old[-1,1:-1],dtype='d'),MPI.DOUBLE],dest=bottom_proc,tag=bottom_send_tag)
    if i_part != 0:
        world.Recv(np.ascontiguousarray(U_local_old[0,1:-1],dtype='d'),source=top_proc,tag=top_recv_tag)
    else:
        col_start_index = (j_part)*(int(global_grid_size[1]/jProcs))
        col_end_index = col_start_index+local_grid_size[1]
        U_local_old[0,1:-1] = BC_top[0,col_start_index:col_end_index].flatten()
    if i_part != iProcs-1:
        world.Recv(np.ascontiguousarray(U_local_old[-1,1:-1],dtype='d'),source=bottom_proc,tag=bottom_recv_tag)
    else:
        col_start_index = (j_part)*(int(global_grid_size[1]/jProcs))
        col_end_index = col_start_index+local_grid_size[1]
        U_local_old[-1,1:-1] = BC_bottom[0,col_start_index:col_end_index].flatten()
    return

step = 1
print "Communicating external boundary conditions to boundary processors"
# communicate_horizontal_BC_from_master(rank,nProcs,jProcs,local_grid_size)
communicate_vertical_BC_from_master(rank,nProcs,jProcs,local_grid_size)

print "Computation starts on processor",rank
while (global_error[0]>tolerance):
    if rank == 0:
        t1 = t.time()
    get_horizontal_boundary_vals(rank,jProcs,U_local_old)
    get_vertical_boundary_vals(rank,jProcs,iProcs,U_local_old)
    U_local_new[1:-1,1:-1] = U_local_old[1:-1,1:-1]+prefactor*(U_local_old[:-2,1:-1]+U_local_old[2:,1:-1]-2*U_local_old[1:-1,1:-1])
    # print(np.amax(np.abs(U_local_new-U_local_old)),rank)
    local_error[0] = np.sum(np.power(np.abs(U_local_new-U_local_old),2))
    U_local_old = np.copy(U_local_new)
    if rank == 0:
        t2 = t.time()
        time_taken = time_taken + t2 - t1
    if step%10 == 0:
        if rank != 0:
            world.Send([np.ascontiguousarray(U_local_old[1:-1,1:-1],dtype='d'),MPI.DOUBLE],dest=0,tag=(100*step)+rank)
        else:
            T_curr = np.zeros(global_grid_size,dtype='d')
            for i_part in range(iProcs):
                for j_part in range(jProcs):
                    proc_id = i_part*jProcs + j_part
                    rank_tag = (100*step)+proc_id
                    row_start_index = i_part*local_grid_size[0]
                    col_start_index = j_part*local_grid_size[1]
                    if i_part != iProcs-1:
                        row_end_index = row_start_index+local_grid_size[0]
                    else:
                        row_end_index = global_grid_size[0]
                    if j_part != jProcs-1:
                        col_end_index = col_start_index+local_grid_size[1]
                    else:
                        col_end_index = global_grid_size[1]
                    if proc_id == 0:
                        T_curr[row_start_index:row_end_index,col_start_index:col_end_index] = U_local_old[1:-1,1:-1]
                    else:
                        world.Recv(np.ascontiguousarray(T_curr[row_start_index:row_end_index,col_start_index:col_end_index],dtype='d'),source=proc_id,tag=(100*step)+proc_id)
        if rank == 0:
            layer = write_to_file_2d(T_curr,layer)
    world.Allreduce(local_error,global_error,op=MPI.MAX)
    step = step + 1
else:
    print "Global tolerance reached"

if rank == 0:
    print("time taken:",time_taken)
    print("steps:",step)
    print("global error",global_error[0])
