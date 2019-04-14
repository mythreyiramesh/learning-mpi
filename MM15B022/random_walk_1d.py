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

# Defining domains for each processor
total_domain_length = 120
if total_domain_length%nProcs != 0:
    print "Domain not evenly divisible, please change!"
    exit(1)

local_domain_length = total_domain_length/nProcs
domain_start = (rank%nProcs) * local_domain_length

# Definiting initial particle size for each processor
total_no_of_particles = 480
if total_no_of_particles%nProcs != 0:
    print "Particles not evenly divisible, please change!"
    exit(1)

local_no_of_particles = np.array([total_no_of_particles/nProcs],dtype='i')
if (rank == 0):
    particles_in_each_proc = np.zeros(nProcs,dtype='i')

list_of_locations = domain_initialisation(local_no_of_particles,local_domain_length,domain_start)
particleList = []
for id in range(local_no_of_particles):
    part_id = rank*local_no_of_particles + id
    particleList.append(Particle(part_id,list_of_locations[id]))

# for i in range(local_no_of_particles):
#     print(particleList[i],rank,i)

# defining some communication protocols
def sendParticle(particle,to_id,send_id):
    world.send(particle,dest=to_id,tag=send_id)

def recvParticle(from_id,recv_id):
    particle = world.recv(source=from_id,tag=recv_id)
    return particle

def deleteParticle(particle_ID,particleList):
    search_index = -1
    for id,particle in enumerate(particleList):
        if particle.pID == particle_ID:
            search_index = id
            break
    particleList.pop(search_index)
    return particleList

def send_left(particlesLeft,step):
    left_neighbour = (rank-1)%nProcs
    # First send the number of particles present in the list
    # Left send and right receive will have series 2000000+iter_no*10000
    # length is communicated in channel 999
    no_of_left_particles = np.array([len(particlesLeft)],dtype='i')
    if no_of_left_particles >= 998:
        print("Find better tag system!")
        exit(1)
    world.Send([no_of_left_particles,MPI.INT],dest=left_neighbour,tag=(2000000+(step*10000)+999))
    for i in range(no_of_left_particles):
        sendParticle(particlesLeft[i],left_neighbour,(2000000+(step*10000)+i))

def receive_right(particles,step):
    right_neighbour = (rank+1)%nProcs
    # First receive the number of particles
    no_of_right_particles = np.array([0],dtype='i')
    world.Recv(no_of_right_particles,source=right_neighbour,tag=(2000000+(step*10000)+999))
    recv_particle = Particle()
    for i in range(no_of_right_particles):
        recv_particle = recvParticle(right_neighbour,(2000000+(step*10000)+i))
        particles.append(recv_particle)
    return particles

def send_right(particlesRight,step):
    right_neighbour = (rank+1)%nProcs
    # First send the number of particles present in the list
    # Left send and right receive will have series 3000000+iter_no*10000
    # length is communicated in channel 999
    no_of_right_particles = np.array([len(particlesRight)],dtype='i')
    if no_of_right_particles >= 998:
        print("Find better tag system!")
        exit(1)
    world.Send([no_of_right_particles,MPI.INT],dest=right_neighbour,tag=(3000000+(step*10000)+999))
    for i in range(no_of_right_particles):
        sendParticle(particlesRight[i],right_neighbour,(3000000+(step*10000)+i))

def receive_left(particles,step):
    left_neighbour = (rank-1)%nProcs
    # First receive the number of particles
    no_of_left_particles = np.array([0],dtype='i')
    world.Recv(no_of_left_particles,source=left_neighbour,tag=(3000000+(step*10000)+999))
    recv_particle = Particle()
    for i in range(no_of_left_particles):
        recv_particle = recvParticle(left_neighbour,(3000000+(step*10000)+i))
        particles.append(recv_particle)
    return particles

def communicate_to_master(step):
    # Let this channel be 10000+(step*100)+rank
    if (rank != 0):
        world.Send(local_no_of_particles,dest=0,tag=(10000+(step*100)+rank))
    else:
        clear_output_file()
        particles_in_each_proc[0] = local_no_of_particles
        for proc_id in range(1,nProcs):
            recv_no = np.array([0],dtype='i')
            world.Recv(recv_no,source=proc_id,tag=((10000+(step*100)+proc_id)))
            particles_in_each_proc[proc_id] = recv_no
        write_to_file_1d(particles_in_each_proc)

# starting random walk
def get_new_location():
    upper_limit = np.floor(local_domain_length/2)
    lower_limit = np.floor(-1*local_domain_length/2)
    return np.floor(np.random.rand(1)*(upper_limit-lower_limit))+lower_limit

def check_new_location(location):
    if location < domain_start:
        return -1
    elif location >= domain_start+local_domain_length:
        return 1
    else:
        return 0

total_no_of_iterations = 1000

for iter_no in range(total_no_of_iterations):
    particlesLeft = []
    particlesRight = []
    for particle in particleList:
        particle.pX = get_new_location()
        if check_new_location(particle.pX) > 0:
            particlesRight.append(particle)
        elif check_new_location(particle.pX) < 0:
            particlesLeft.append(particle)
    # For this tag system, basic assumptions are: the number of processors doesn't exceed 99, the number of particles do not exceed 98
    # Left send and right receive will have series 20000+iter_no*100
    # Right send and right receive will have 30000+iter_no*100
    send_left(particlesLeft,iter_no)
    # send_right(particlesRight,iter_no)
    # receive_left(particleList,iter_no)
    particleList = receive_right(particleList,iter_no)
    for i in range(len(particlesLeft)):
        particleList = deleteParticle(particlesLeft[i].pID,particleList)
    for i in range(len(particlesRight)):
        particleList = deleteParticle(particlesRight[i].pID,particleList)
    # print("Length of particleList after iteration",iter_no,"in processor",rank,"is",local_no_of_particles)
    # Let this channel be 10000+(step*100)+rank
    local_no_of_particles = np.array([len(particleList)],dtype='i')
    communicate_to_master(iter_no)
    # if (rank==0):
    #     print iter_no,particles_in_each_proc,sum(particles_in_each_proc)

# print "Done",rank
# if (rank==0):
#     part0 = Particle(0,5)
#     print("before",rank,part0)
#     sendParticle(part0,1,100)
#     print("after",rank,part0)
# else:
#     part1 = Particle()
#     print("before",rank,part1)
#     part1 = recvParticle(0,100)
#     print("after",rank,part1)
