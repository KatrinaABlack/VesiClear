from math import sqrt
import numpy as np

def downsample_contour(particles, dist, psize):
    # Select a sparse set of particles from a contour such that each particle
    # is at least dist nm from its previous neighbor
    # Precondition: particles are sorted by angle (clockwise or anticlockwise)
    particles_downsampled = [particles[0]]
    for particle in particles:
        if np.linalg.norm(particle - particles_downsampled[-1]) * psize >= dist:
            particles_downsampled.append(particle)
    return particles_downsampled


def proj_dist(p, d):
    # Compute length of projection of p onto d
    # ERROR HANDLING: Prevent division by zero
    d_mag = sqrt(d[0] ** 2 + d[1] ** 2)
    if d_mag < 1e-10:
        return 0.0
    return (p[0] * d[0] + p[1] * d[1]) / d_mag