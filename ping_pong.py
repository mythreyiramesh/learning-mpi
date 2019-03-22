from mpi4py import MPI
import numpy as np

world = MPI.COMM_WORLD
rank = world.Get_rank()

MAX_PINGS_AND_PONGS = 50

count = np.array([0], dtype='i')
partner_rank = (rank+1)%2

while count[0] < MAX_PINGS_AND_PONGS:
    # if count is even, rank 0 must send and if count is odd, rank 1 must send
    if (count[0]%2 == rank):
        count[0] = count[0]+1
        world.Send([count,MPI.INT],dest=partner_rank,tag=1)
        print(rank,'sent count',count[0],'to',partner_rank)
    else:
        world.Recv([count,MPI.INT],source=partner_rank)
        print(rank,'received count',count[0],'from',partner_rank)
