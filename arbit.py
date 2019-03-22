from mpi4py import MPI
import numpy as np
from math import floor

# assuming that the code is going to be run only with 2 procs.
world = MPI.COMM_WORLD
rank = world.Get_rank()
nprocs = world.Get_size()
if(nprocs != 2):
    print("Run on 2 procs only")
    # exit?


if (rank == 0):
    length = np.array([floor(np.random.random(1)*10)],dtype='i') # this is a random length.
    print "Today's random size is: ",length[0]

    world.Send([length,MPI.INT],dest=1,tag=1)
    a = np.arange(length,dtype='i')
    world.Send([a,MPI.INT],dest=1, tag=2)

if (rank != 0):

    rand_length = np.array([0],dtype='i')

    world.Recv([rand_length,MPI.INT],source=0,tag=1)
    print("Received info that today's length is",rand_length[0])

    data = np.arange(rand_length,dtype='i')
    world.Recv([data,MPI.INT],source=0,tag=2)
    print(data)
