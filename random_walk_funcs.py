import numpy as np

def domain_initialisation(no_of_particles,domain_size,domain_start):
    relative_positions = np.floor(domain_size*np.random.rand(no_of_particles))
    absolute_positions = domain_start*np.ones(no_of_particles)+relative_positions
    return absolute_positions
