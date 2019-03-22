from mpi4py import MPI
import numpy as M
import time as TIME

Comm 	= MPI.COMM_WORLD;
NumProc = Comm.Get_size();

T 		= M.array([-1], dtype = 'i');

Rank 	 = Comm.Get_rank();
PrevMate = (Rank-1)%NumProc
NextMate = (Rank+1)%NumProc;

if Rank == 0:

	T+=1;

	Comm.Send([T,MPI.INT], dest  = NextMate, tag = 11);
	print T,"Send from process  ", Rank, "to", NextMate

	Comm.Recv([T,MPI.INT],source = PrevMate, tag = 11);
	print T,"Recieved on process", Rank, "from", PrevMate

else:
	Comm.Recv([T,MPI.INT],source = PrevMate, tag = 11);
	print T,"Recieved on process", Rank, "from", PrevMate

	T+=1;

	Comm.Send([T,MPI.INT], dest  = NextMate, tag = 11);
	print T,"Send from process  ", Rank, "to", NextMate
