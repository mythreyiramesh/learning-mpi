from mpi4py import MPI
import numpy as np

world = MPI.COMM_WORLD
rank = world.Get_rank()
size = world.Get_size()

count = np.array([-1], dtype='i')
next_rank = (rank+1)%size
prev_rank = (rank-1)%size
#print("Rank",rank,"next",next_rank,"prev",prev_rank)

if (rank!=0):
    world.Recv([count,MPI.INT],source=prev_rank,tag=1)
    print(rank,'received count',count[0],'from',prev_rank)
    world.Send([count,MPI.INT],dest=next_rank,tag=1)
    print(rank,'sent count',count[0],'to',next_rank)
else:
    world.Send([count,MPI.INT],dest=next_rank,tag=1)
    print(rank,'sent count',count[0],'to',next_rank)
    world.Recv([count,MPI.INT],source=prev_rank,tag=1)
    print(rank,'received count',count[0],'from',prev_rank)
