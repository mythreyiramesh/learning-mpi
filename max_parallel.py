from mpi4py import MPI
import numpy as np

world = MPI.COMM_WORLD
rank = world.Get_rank()
nProcs = world.Get_size() # for now, this will be 4
local_size = np.array([0],dtype='d')

if (rank == 0):
    N = np.random.rand(120000000)
    lims = np.linspace(0,len(N),nProcs+1)
    # print(lims)
    local_size = len(N)/nProcs # Note: this must be checked

local_size = world.bcast(local_size,root=0)
# print(local_size)
N_local = np.zeros(local_size,dtype='d')

if (rank == 0):
    # print(N)
    N_local = N[lims[0]:lims[1]]
    for proc_id in range(1,nProcs):
        rank_tag = 100 + proc_id
        world.Send([N[lims[proc_id]:lims[proc_id+1]],MPI.DOUBLE],dest=proc_id,tag=rank_tag)
        #print(np.size(N[lims[proc_id]:lims[proc_id+1]]))
else:
    world.Recv(N_local,source=0,tag=100+rank)

max_local = np.amax(N_local)
max_global = np.array([0],dtype='d')
# print(N_local,rank)
if (rank != 0):
    rank_tag = 200 + rank
    world.Send([max_local,MPI.DOUBLE],dest=0,tag=rank_tag)
else:
    max_array = np.zeros(nProcs,dtype='d')
    recv_temp = np.zeros(1,dtype='d')
    for proc_id in range(1,nProcs):
        world.Recv(recv_temp,source=proc_id,tag=200+proc_id)
        max_array[proc_id] = recv_temp
    max_array[0] = max_local
    max_global = np.amax(max_array)

max_global = world.bcast(max_global,root=0)
print(max_global,rank)
