from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

data = np.ones(3,dtype=np.float64) # don't put the rank there xD
print(data)

if rank == 0:
    comm.Send([data,MPI.DOUBLE],dest=1,tag=1)
if rank == 1:
    info = MPI.Status()
    comm.Recv([data,MPI.DOUBLE],MPI.ANY_SOURCE,MPI.ANY_TAG,info)
    source = info.Get_source()
    tag = info.Get_tag()
    count = info.Get_elements(MPI.DOUBLE)
    size = info.Get_count()
    print("on",rank,"source, tag, count, size, data is", source, tag, count, size, data)
