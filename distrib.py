from mpi4py import MPI
import numpy as np
from math import floor

world = MPI.COMM_WORLD
rank = world.Get_rank()
nprocs = world.Get_size()

length = 100

# slice_length = floor(length/(nprocs-1)); last_slice will be "till the end" (possible using "rank == nprocs-1")
slice_length = floor(length/(nprocs-1))


# initialise only when rank is zero, because that's where you're sending
if (rank == 0):

    a = np.arange(length,dtype='i')
    # print(a);
    # a = np.zeros(length)

    # rank 1 - 0 to slice_length-1
    # rank 2 - slice_length to 2*slice_length-1
    # ...
    # rank i - [((i-1)*slice_length):(i)*slice_length]
    # last rank - [((i-1)*slice_length):]

    # note! This assumes that the rank is greater than 1
    for i in range(1,nprocs):
        rank_tag = 100 + i # just a rough tag, can be anything
        start_index = (i-1)*slice_length
        end_index = length
        if (i != nprocs-1):
            end_index = i*slice_length
        world.Send([a[start_index:end_index],MPI.INT],dest=i, tag=rank_tag)
        # print(a[start_index:end_index])
        print(rank,'sent',start_index,'to',end_index,'towards rank',i)
        print("that is an array of size",end_index-start_index,"for rank",i)

if (rank!=0):
    if (rank != nprocs-1):
        #temp_array = np.array(slice_length,dtype='i')
        temp_array = np.arange(slice_length,dtype='i')
    else:
        #temp_array = np.array(length-((nprocs-2)*slice_length),dtype='i')
        temp_array = np.arange(length-((nprocs-2)*slice_length),dtype='i')
    print("initialised array of length",np.size(temp_array),"for rank",rank)
    # print(temp_array)

    world.Recv([temp_array,MPI.INT],source=0,tag=(100+rank))
    print(rank,'received',temp_array,'from',0)
