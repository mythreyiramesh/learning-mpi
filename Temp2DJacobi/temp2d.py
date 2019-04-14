from mpi4py import MPI
import numpy as np
from temp2d_funcs import *
import matplotlib.pyplot as plt
import time as t

world = MPI.COMM_WORLD
rank = world.Get_rank()
nProcs = world.Get_size()
iProcs = np.zeros(1,dtype='i')
jProcs = np.zeros(1,dtype='i')

global_grid_size = np.array([0,0],dtype='i'); # arbitrary initialisation
local_grid_size = np.array([0,0],dtype='i'); # arbitrary initialisation
tolerance = 10**(-4) # tolerance

global_error = np.array([10**5],dtype='d'); # arbitrary large number
local_error = np.array([10**5],dtype='d'); # arbitrary large number

if rank == 0:
    domain_width = 100
    domain_height = 100
    T = T_2Dinit([domain_height,domain_width])
    BC_left = np.zeros((domain_height,1),dtype='d');
    BC_right = np.zeros((domain_height,1),dtype='d');
    BC_top = np.zeros((1,domain_width),dtype='d');
    BC_bottom = np.zeros((1,domain_width),dtype='d');
    time_taken = 0;
    layer = 1
    clear_output_file()
    layer = write_to_file_2d(T,layer)
    global_grid_size = np.shape(T);
    # print(global_grid_size)
    iProcs,jProcs = factor_procs(nProcs,global_grid_size)
    # print(iProcs,jProcs)
    # print(type(jProcs))
    local_grid_size = np.array([int(global_grid_size[0]/iProcs),int(global_grid_size[1]/jProcs)],dtype='i')

    for proc_id in range(nProcs):
        # print("proc",proc_id)
        i_part =  int(proc_id/jProcs)
        j_part = proc_id%jProcs
        # print("i,j",i_part,j_part)
        to_send = np.copy(local_grid_size)
        if i_part != iProcs-1 and j_part != jProcs-1:
            world.Send([local_grid_size,MPI.INT],dest=proc_id,tag=(100+proc_id))
            # print("not last in any direction, just sending")
        else:
            if i_part == iProcs-1:
                # print("last row proc, change i limit to")
                if global_grid_size[0]%iProcs == 0:
                    to_send[0] = local_grid_size[0]
                    # print(to_send[0])
                else:
                    to_send[0] = global_grid_size[0]-(iProcs-1)*local_grid_size[0]
                    # print(to_send[0])
            if j_part == jProcs-1:
                # print("last col proc, change j limit to")
                if global_grid_size[0]%jProcs == 0:
                    to_send[1] = local_grid_size[1]
                    # print(to_send[1])
                else:
                    to_send[1] = global_grid_size[1]-(jProcs-1)*local_grid_size[1]
                    # print(to_send[1])
            # print(to_send,proc_id)
            world.Send([to_send,MPI.INT],dest=proc_id,tag=(100+proc_id))
    # if global_grid_size[0]%iProcs == 0:
    #     world.Send([local_grid_size,MPI.DOUBLE],dest=nProcs-1,tag=(100+nProcs-1))
    # else:
    #     world.Send([(global_grid_size[0]-(iProcs-1)*local_grid_size[0]),MPI.DOUBLE],dest=nProcs-1,tag=(100+nProcs-1))
else:
    world.Recv(local_grid_size,source=0,tag=(100+rank))

global_grid_size = world.bcast(global_grid_size,root=0)
iProcs = world.bcast(iProcs,root=0)
jProcs = world.bcast(jProcs,root=0)
# print(iProcs,jProcs,rank)
# print("global",global_grid_size,rank)
# print("local",local_grid_size,rank)

# we are decomposing the domain to nProcs
T_local_prev = np.zeros(local_grid_size+2,dtype='d');
# print(np.shape(T_local_prev),rank,"after init")
T_local_curr = np.zeros(local_grid_size+2,dtype='d');
# T_local_next = np.zeros(local_grid_size); # we don't need this, we are only using m for m+1
# boundaries = np.array([0,0],dtype='d')
# we'll just be storing on the ghosts.

# This whole thing can be eliminated if we're going to initialise randomly
if rank == 0:
    for i_part in range(iProcs):
        for j_part in range(jProcs):
            # print(i_part,j_part,"test")
            proc_id = i_part*jProcs + j_part
            # print("at proc_id",proc_id)
            rank_tag = 200 + proc_id
            row_start_index = i_part*local_grid_size[0]
            col_start_index = j_part*local_grid_size[1]
            # print("Here #1!")
            if i_part != iProcs-1:
                row_end_index = row_start_index+local_grid_size[0]
            else:
                row_end_index = global_grid_size[0]
            # print("Here #2!")
            if j_part != jProcs-1:
                col_end_index = col_start_index+local_grid_size[1]
            else:
                col_end_index = global_grid_size[1]
            # print("Here #3!")
            if proc_id == 0:
                T_local_prev[1:-1,1:-1] = T[row_start_index:row_end_index,col_start_index:col_end_index]
            else:
                # print("sent",np.shape(np.ascontiguousarray(T[row_start_index:row_end_index,col_start_index:col_end_index],dtype='d')),"to",proc_id)
                world.Send([np.ascontiguousarray(T[row_start_index:row_end_index,col_start_index:col_end_index],dtype='d'),MPI.DOUBLE],dest=proc_id,tag=rank_tag)
else:
    # print("receiving",np.shape(T_local_prev[1:-1,1:-1]),"at",rank)
    world.Recv(np.ascontiguousarray(T_local_prev[1:-1,1:-1],dtype='d'),source=0,tag=(200+rank))

# print(T_local_prev)
# print("done initialisation",rank)

def communicate_horizontal_BC_from_master(rank,nProcs,jProcs,local_grid_size):
    # Special channel to communicate right boundary values: 450
    # Special channel to communicate left boundary values: 350
    # print(type(jProcs),"at rank",rank)
    # print("arrived at",rank)
    if rank == 0:
        global BC_right, BC_left
        for proc_id in range(1,nProcs):
            if proc_id%jProcs == jProcs-1:
                world.Send([BC_right,MPI.DOUBLE],dest=proc_id,tag=450+proc_id)
                # print("sent right to",proc_id)
            if proc_id%jProcs == 0:
                world.Send([BC_left,MPI.DOUBLE],dest=proc_id,tag=350+proc_id)
                # print("sent left to",proc_id)
    else:
        # print(type(rank),type(jProcs))
        if rank%jProcs == jProcs-1:
            BC_right = np.zeros((global_grid_size[0],1),dtype='d')
            world.Recv(BC_right,source=0,tag=450+rank)
            # print("receiving right at",rank)
        if rank%jProcs == 0:
            BC_left = np.zeros((global_grid_size[0],1),dtype='d')
            world.Recv(BC_left,source=0,tag=350+rank)
            # print("receiving left at",rank)
    return

def communicate_vertical_BC_from_master(rank,nProcs,jProcs,local_grid_size):
    # Special channel to communicate top boundary values: 550
    # Special channel to communicate bottom boundary values: 650
    # print(type(jProcs),"at rank",rank)
    # print("arrived at",rank)
    if rank == 0:
        global BC_bottom,BC_top
        for proc_id in range(1,nProcs):
            if int(proc_id/jProcs) == iProcs-1:
                world.Send([BC_bottom,MPI.DOUBLE],dest=proc_id,tag=650+proc_id)
                # print("sent bottom to",proc_id,np.shape(BC_top))
            if int(proc_id/jProcs) == 0:
                world.Send([BC_top,MPI.DOUBLE],dest=proc_id,tag=550+proc_id)
                # print("sent top to",proc_id,np.shape(BC_top))
    else:
        # print(type(rank),type(jProcs))
        if int(rank/jProcs) == iProcs-1:
            BC_bottom = np.zeros((1,global_grid_size[1]),dtype='d')
            world.Recv(BC_bottom,source=0,tag=650+rank)
            # print("receiving bottom at",rank)
        if int(rank/jProcs) == 0:
            BC_top = np.zeros((1,global_grid_size[1]),dtype='d')
            # print("receiving at",rank,np.shape(BC_top))
            world.Recv(BC_top,source=0,tag=550+rank)
            # print("receiving top at",rank)
    return

def get_horizontal_boundary_vals(rank,jProcs,T_local_prev):
    # bound_left = np.zeros((local_grid_size[0],1),dtype='d')
    # bound_right = np.zeros((local_grid_size[0],1),dtype='d')
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
    # print("At rank",rank,"sending left value to",left_proc,"with tag",left_send_tag)
    if j_part != 0:
        world.Send([np.ascontiguousarray(T_local_prev[1:-1,1],dtype='d'),MPI.DOUBLE],dest=left_proc,tag=left_send_tag)
    if j_part != jProcs-1:
        world.Send([np.ascontiguousarray(T_local_prev[1:-1,-1],dtype='d'),MPI.DOUBLE],dest=right_proc,tag=right_send_tag)
    # print("At rank",rank,"receiving from",left_proc,"with tag",left_recv_tag)
    if j_part != 0:
        world.Recv(np.ascontiguousarray(T_local_prev[1:-1,0],dtype='d'),source=left_proc,tag=left_recv_tag)
    else:
        row_start_index = (i_part)*(int(global_grid_size[0])/iProcs)
        # col_start_index = (j_part)*(int(global_grid_size[1])/jProcs)
        row_end_index = row_start_index+local_grid_size[0]
        # col_end_index = col_start_index+local_grid_size[1]
        # print(np.shape(T_local_prev[1:-1,0]),np.shape(BC_left[row_start_index:row_end_index]),rank)
        T_local_prev[1:-1,0] = BC_left[row_start_index:row_end_index].flatten()
    if j_part != jProcs-1:
        world.Recv(np.ascontiguousarray(T_local_prev[1:-1,-1],dtype='d'),source=right_proc,tag=right_recv_tag)
    else:
        global BC_right
        row_start_index = (i_part)*(int(global_grid_size[0]/iProcs))
        row_end_index = row_start_index+local_grid_size[0]
        # print(np.shape(T_local_prev[1:-1,-1]),np.shape(BC_right[row_start_index:row_end_index]),rank)
        T_local_prev[1:-1,-1] = BC_right[row_start_index:row_end_index].flatten()
    return

def get_vertical_boundary_vals(rank,jProcs,iProcs,T_local_prev):
    # bound_left = np.zeros((local_grid_size[0],1),dtype='d')
    # bound_right = np.zeros((local_grid_size[0],1),dtype='d')
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
    # print("At rank",rank,"sending left value to",left_proc,"with tag",left_send_tag)
    if i_part != 0:
        world.Send([np.ascontiguousarray(T_local_prev[1,1:-1],dtype='d'),MPI.DOUBLE],dest=top_proc,tag=top_send_tag)
    if i_part != iProcs-1:
        world.Send([np.ascontiguousarray(T_local_prev[-1,1:-1],dtype='d'),MPI.DOUBLE],dest=bottom_proc,tag=bottom_send_tag)
    # print("At rank",rank,"receiving from",left_proc,"with tag",left_recv_tag)
    if i_part != 0:
        world.Recv(np.ascontiguousarray(T_local_prev[0,1:-1],dtype='d'),source=top_proc,tag=top_recv_tag)
    else:
        # row_start_index = (i_part)*(int(global_grid_size[0])/iProcs)
        col_start_index = (j_part)*(int(global_grid_size[1]/jProcs))
        # print(col_start_index,rank)
        # row_end_index = row_start_index+local_grid_size[0]
        col_end_index = col_start_index+local_grid_size[1]
        # print(col_end_index,rank)
        # print(np.shape(T_local_prev[0,1:-1]),np.shape(BC_top[0,col_start_index:col_end_index].flatten()),rank)
        T_local_prev[0,1:-1] = BC_top[0,col_start_index:col_end_index].flatten()
    if i_part != iProcs-1:
        world.Recv(np.ascontiguousarray(T_local_prev[-1,1:-1],dtype='d'),source=bottom_proc,tag=bottom_recv_tag)
    else:
        # global BC_bottom
        col_start_index = (j_part)*(int(global_grid_size[1]/jProcs))
        col_end_index = col_start_index+local_grid_size[1]
        # print(np.shape(T_local_prev[-1,1:-1]),np.shape(BC_bottom[col_start_index:col_end_index]),rank)
        T_local_prev[-1,1:-1] = BC_bottom[0,col_start_index:col_end_index].flatten()
    return

# if rank == 0:
#     print(T_local_curr)
# plt.plot(T_local_prev)

# total_steps = 100;
# first let it perform calculations for all the inside nodes, this is independent of the processor.
step = 1
communicate_horizontal_BC_from_master(rank,nProcs,jProcs,local_grid_size)
communicate_vertical_BC_from_master(rank,nProcs,jProcs,local_grid_size)

while (global_error[0]>tolerance):
    # print("step",step,rank)
    if rank == 0:
        t1 = t.time()
    # print(local_grid_size,rank)
    # print(type(jProcs),"before call",rank)
    # print(np.shape(T_local_prev),rank,step,"before call")
    get_horizontal_boundary_vals(rank,jProcs,T_local_prev)
    get_vertical_boundary_vals(rank,jProcs,iProcs,T_local_prev)
    # print("done",rank)
    ####
    # print(T_local_curr,"curr before",step,rank)
    # print(T_local_prev,"prev before",step,rank)
    # print(T_local_curr is T_local_prev)
    # T_local_curr[0] = 0.5*(boundaries[0]+T_local_prev[1])
    # print(T_local_curr is T_local_prev)
    # for i in range(1,local_grid_size-1):
    #     T_local_curr[i] = 0.5*(T_local_prev[i-1] + T_local_prev[i+1])
    # T_local_curr[1:-1] = 0.5*(T_local_prev[:-2]+T_local_prev[2:])
    # print(T_local_curr is T_local_prev)
    # T_local_curr[-1] = 0.5*(boundaries[1]+T_local_prev[-2])
    # print(T_local_curr is T_local_prev)
    # print(T_local_prev,"prev after",step,rank)
    # print(T_local_curr,"curr after",step,rank)
    # print(np.amax(np.abs(T_local_curr-T_local_prev)),rank)
    T_local_curr[1:-1,1:-1] = 0.25*(T_local_prev[:-2,1:-1]+T_local_prev[2:,1:-1]+T_local_prev[1:-1,:-2]+T_local_prev[1:-1,2:])
    local_error[0] = np.sum(np.power(np.abs(T_local_curr-T_local_prev),2))
    T_local_prev = np.copy(T_local_curr)
    if rank == 0:
        t2 = t.time()
        time_taken = time_taken + t2 - t1
    if step%10 == 0:
        # T_curr = np.zeros(global_grid_size,dtype='d')
        if rank != 0:
            world.Send([np.ascontiguousarray(T_local_prev[1:-1,1:-1],dtype='d'),MPI.DOUBLE],dest=0,tag=(100*step)+rank)
        else:
            T_curr = np.zeros(global_grid_size,dtype='d')
            for i_part in range(iProcs):
                for j_part in range(jProcs):
                    # print(i_part,j_part,"test")
                    proc_id = i_part*jProcs + j_part
                    # print("at proc_id",proc_id)
                    rank_tag = (100*step)+proc_id
                    row_start_index = i_part*local_grid_size[0]
                    col_start_index = j_part*local_grid_size[1]
                    # print("Here #1!")
                    if i_part != iProcs-1:
                        row_end_index = row_start_index+local_grid_size[0]
                    else:
                        row_end_index = global_grid_size[0]
                    # print("Here #2!")
                    if j_part != jProcs-1:
                        col_end_index = col_start_index+local_grid_size[1]
                    else:
                        col_end_index = global_grid_size[1]
                    if proc_id == 0:
                        T_curr[row_start_index:row_end_index,col_start_index:col_end_index] = T_local_prev[1:-1,1:-1]
                    else:
                        world.Recv(np.ascontiguousarray(T_curr[row_start_index:row_end_index,col_start_index:col_end_index],dtype='d'),source=proc_id,tag=(100*step)+proc_id)
        # world.Gather(T_local_prev,T_curr,root=0)
        if rank == 0:
            layer = write_to_file_2d(T_curr,layer)
    # print("local",local_error,"global",global_error,"rank",rank)
    world.Allreduce(local_error,global_error,op=MPI.MAX)
    # print(global_error,rank)
    step = step + 1
    ####
    # if rank == 0:
    #     print T_local_prev
    # step = step+1


# print(T_local_curr)

# plt.plot(T_local_curr)
# plt.show()
if rank == 0:
    print("time taken:",time_taken)
    print("steps:",step)
    print("global error",global_error[0])
# print("Done",rank)
