from mpi4py import MPI
import numpy as np
from random_walk_funcs import *

world = MPI.COMM_WORLD
rank = world.Get_rank()
nProcs = world.Get_size()

class Particle:
    pID = None
    pX = 0
    def __init__(self,pID=None,pX=0):
        self.pID = pID
        self.pX = pX
    def __repr__(self):
        return "ID: %s; Location: %s" % (self.pID, self.pX)

total_domain_length = 100
if total_domain_length%nProcs != 0:
    print "Domain not evenly divisible, please change!"
    exit(1)

local_domain_length = total_domain_length/nProcs
domain_start = (rank%nProcs) * local_domain_length

total_no_of_particles = 12
if total_no_of_particles%nProcs != 0:
    print "Particles not evenly divisible, please change!"
    exit(1)

local_no_of_particles = total_no_of_particles/nProcs

list_of_locations = domain_initialisation(local_no_of_particles,local_domain_length,domain_start)
particleList = []
for id in range(local_no_of_particles):
    part_id = rank*local_no_of_particles + id
    particleList.append(Particle(part_id,list_of_locations[id]))

for i in range(local_no_of_particles):
    print(particleList[i],rank,i)

# defining some communication protocols
def sendParticle(particle,to_id,send_id):
    world.send(particle,dest=to_id,tag=send_id)

def recvParticle(from_id,recv_id):
    particle = world.recv(source=from_id,tag=recv_id)
    return particle

if (rank==0):
    part0 = Particle(0,5)
    print("before",rank,part0)
    sendParticle(part0,1,100)
    print("after",rank,part0)
else:
    part1 = Particle()
    print("before",rank,part1)
    part1 = recvParticle(0,100)
    print("after",rank,part1)
