import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

def figure_eight(t):
    x = np.sin(t)
    y = np.sin(t) * np.cos(t)
    return np.array([x, y])

def motion_model(particles, dt):
    particles[:, 0] += particles[:, 2] * dt + np.random.normal(0, 0.1, particles.shape[0])  # x position
    particles[:, 1] += particles[:, 3] * dt + np.random.normal(0, 0.1, particles.shape[0])  # y position
    particles[:, 2] += np.random.normal(0, 0.05, particles.shape[0]) # x velocity
    particles[:, 3] += np.random.normal(0, 0.05, particles.shape[0]) # y velocity
    return particles

def measurement_model(particle, measurement):
    distance = np.linalg.norm(particle[:2] - measurement)
    return np.exp(-distance**2 / 0.1) # Gaussian chance

def particle_filter(particles, weights, measurement, dt):
    particles = motion_model(particles, dt)
    weights = np.array([measurement_model(p, measurement) for p in particles])
    weights /= np.sum(weights)  # Normalize the weights

    # Resampling but w/ a simple resampling wheel
    N = len(particles)
    indices = np.random.choice(N, size=N, p=weights)
    particles = particles[indices]
    weights = np.ones(N) / N # reset weights after resampling. something your mother wished she could do
    return particles, weights

# parameters for the stuff
num_particles = 1000
dt = 0.1
total_time = 200
time_steps = int(total_time / dt)

# Initialize particles n weights n shit
particles = np.zeros((num_particles, 4)) # [x, y, vx, vy]
particles[:, 0] = np.random.uniform(-1.5, 1.5, num_particles)
particles[:, 1] = np.random.uniform(-1.5, 1.5, num_particles)
particles[:, 2] = np.random.uniform(-0.5, 0.5, num_particles)
particles[:, 3] = np.random.uniform(-0.5, 0.5, num_particles)

weights = np.ones(num_particles) / num_particles

# generate smegma points
true_trajectory = [figure_eight(t * dt) for t in range(time_steps)]
measurements = [point + np.random.normal(0, 0.2, 2) for point in true_trajectory] # Add noise to measurements

# plotting
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-2, 2)
ax.set_ylim(-1.5, 1.5)
true_line, = ax.plot([], [], label="True Trajectory", color="blue")
meas_scatter = ax.scatter([], [], label="Measurements", color="red", s=5)
est_line, = ax.plot([], [], label="Estimated Trajectory", color="green")
particle_scatter = ax.scatter(particles[:, 0], particles[:, 1], s=1, color="gray", alpha=0.5)
ax.legend()
ax.set_title("Particle Filter Tracking Figure 8")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.grid(True)

def animate(i):
    global particles, weights
    measurement = measurements[i]
    particles, weights = particle_filter(particles, weights, measurement, dt)
    estimated_position = np.average(particles[:, :2], axis=0, weights=weights)

    true_line.set_data([p[0] for p in true_trajectory[:i+1]], [p[1] for p in true_trajectory[:i+1]])
    meas_scatter.set_offsets(measurements[:i+1])
    est_line.set_data([estimated_position[0]],[estimated_position[1]])
    particle_scatter.set_offsets(particles[:, :2])
    return true_line, meas_scatter, est_line, particle_scatter,

ani = animation.FuncAnimation(fig, animate, frames=time_steps, interval=50, blit=True)
plt.show()
