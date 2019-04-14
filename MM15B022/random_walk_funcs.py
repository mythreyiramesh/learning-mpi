import numpy as np

def domain_initialisation(no_of_particles,domain_size,domain_start):
#    print(type(no_of_particles))
#    print(np.size(no_of_particles))
    relative_positions = np.floor(domain_size*np.random.rand(no_of_particles[0]))
    absolute_positions = domain_start*np.ones(no_of_particles)+relative_positions
    return absolute_positions

def clear_output_file():
    f = file("Particles.m","w")
    line = ""
    f.close()

def write_to_file_1d(particles):
    size = np.shape(particles)[0]
    line = "P = ["
    for i in range(size):
        line = line + str(float(particles[i])) + " "
    line = line + "];\n"
    f = file("Particles.m","a")
    f.writelines(line)
    f.close()
    return
