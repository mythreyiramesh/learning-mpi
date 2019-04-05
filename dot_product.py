from mpi4py import MPI
import numpy as np


world = MPI.COMM_WORLD
rank = world.Get_rank()
nProcs = world.Get_size()

N = 10000;

local_size = N/nProcs;
local_a = np.zeros(local_size,dtype='d')
local_b = np.zeros(local_size,dtype='d')

if rank==0:
    a = np.arange(N,dtype='d')
    b = np.arange(N,dtype='d')

    lims = np.linspace(0,N,nProcs+1)

    local_a = a[lims[0]:lims[1]]
    local_b = b[lims[0]:lims[1]]

    for proc_id in range(1,nProcs):
        send_tag_a = 100+proc_id
        send_tag_b = 200+proc_id
        world.Send([a[lims[proc_id]:lims[proc_id+1]],MPI.DOUBLE],dest=proc_id,tag=send_tag_a)
        world.Send([b[lims[proc_id]:lims[proc_id+1]],MPI.DOUBLE],dest=proc_id,tag=send_tag_b)
else:
    world.Recv(local_a,source=0,tag=100+rank)
    world.Recv(local_b,source=0,tag=200+rank)

local_ans = np.dot(np.transpose(local_a),local_b)
global_ans = np.array([0],dtype='d')
world.Allreduce(local_ans,global_ans,op=MPI.SUM)

print(global_ans,rank)
