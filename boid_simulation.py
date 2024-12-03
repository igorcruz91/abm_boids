import tkinter as tk
import random
import math
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Global variable to control the display of the perception radius
show_radius = True

# Vector class for 2D vector operations
class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def copy(self):
        return Vector(self.x, self.y)

    def add(self, v):
        self.x += v.x
        self.y += v.y

    def sub(self, v):
        self.x -= v.x
        self.y -= v.y

    def mult(self, n):
        self.x *= n
        self.y *= n

    def div(self, n):
        if n != 0:
            self.x /= n
            self.y /= n

    def mag(self):
        return math.hypot(self.x, self.y)

    def normalize(self):
        m = self.mag()
        if m != 0:
            self.div(m)

    def limit(self, max_val):
        if self.mag() > max_val:
            self.normalize()
            self.mult(max_val)

    def distance(self, v):
        return math.hypot(self.x - v.x, self.y - v.y)

    def set_mag(self, n):
        self.normalize()
        self.mult(n)

    def heading(self):
        return math.atan2(self.y, self.x)

# Boid class implementing the behaviors
class Boid:
    def __init__(self, x, y, width, height):
        self.position = Vector(x, y)
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = Vector(math.cos(angle), math.sin(angle))
        self.velocity.mult(random.uniform(1, 3))
        self.acceleration = Vector()
        self.max_force = 0.05
        self.max_speed = 2
        self.perception_radius = 50
        self.width = width
        self.height = height

    def edges(self):
        # Wrap around the screen edges
        if self.position.x > self.width:
            self.position.x = 0
        elif self.position.x < 0:
            self.position.x = self.width

        if self.position.y > self.height:
            self.position.y = 0
        elif self.position.y < 0:
            self.position.y = self.height

    def align(self, boids):
        steering = Vector()
        total = 0
        avg_vector = Vector()
        for other in boids:
            distance = self.position.distance(other.position)
            if other != self and distance < self.perception_radius:
                avg_vector.add(other.velocity)
                total += 1
        if total > 0:
            avg_vector.div(total)
            avg_vector.set_mag(self.max_speed)
            steering = avg_vector.copy()
            steering.sub(self.velocity)
            steering.limit(self.max_force)
        return steering

    def cohesion(self, boids):
        steering = Vector()
        total = 0
        center_of_mass = Vector()
        for other in boids:
            distance = self.position.distance(other.position)
            if other != self and distance < self.perception_radius:
                center_of_mass.add(other.position)
                total += 1
        if total > 0:
            center_of_mass.div(total)
            vector_to_com = center_of_mass.copy()
            vector_to_com.sub(self.position)
            vector_to_com.set_mag(self.max_speed)
            steering = vector_to_com.copy()
            steering.sub(self.velocity)
            steering.limit(self.max_force)
        return steering

    def separation(self, boids):
        steering = Vector()
        total = 0
        avg_vector = Vector()
        for other in boids:
            distance = self.position.distance(other.position)
            if other != self and distance < self.perception_radius / 2 and distance > 0:
                diff = self.position.copy()
                diff.sub(other.position)
                diff.div(distance)
                avg_vector.add(diff)
                total += 1
        if total > 0:
            avg_vector.div(total)
            avg_vector.set_mag(self.max_speed)
            steering = avg_vector.copy()
            steering.sub(self.velocity)
            steering.limit(self.max_force)
        return steering

    def flock(self, boids, align_weight, cohesion_weight, separation_weight):
        alignment = self.align(boids)
        cohesion = self.cohesion(boids)
        separation = self.separation(boids)

        alignment.mult(align_weight)
        cohesion.mult(cohesion_weight)
        separation.mult(separation_weight)

        self.acceleration.add(alignment)
        self.acceleration.add(cohesion)
        self.acceleration.add(separation)

    def update(self):
        self.position.add(self.velocity)
        self.velocity.add(self.acceleration)
        self.velocity.limit(self.max_speed)
        self.acceleration.mult(0)

# Main Application
def main():
    root = tk.Tk()
    root.title("Boid Simulation with Cluster Detection")

    WIDTH = 800
    HEIGHT = 600

    # Global variable to control the display of the perception radius
    global show_radius
    show_radius = True

    # Create main layout frames
    simulation_frame = tk.Frame(root)
    simulation_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    control_frame = tk.Frame(root)
    control_frame.pack(side=tk.LEFT, fill=tk.Y)

    graph_frame = tk.Frame(root)
    graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Create canvas for boid simulation in the simulation frame
    canvas = tk.Canvas(simulation_frame, width=WIDTH, height=HEIGHT, bg="white")
    canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Control frame sliders and buttons
    max_speed_label = tk.Label(control_frame, text="Max Speed")
    max_speed_label.pack()
    max_speed_scale = tk.Scale(control_frame, from_=0, to=10, resolution=0.1, orient=tk.HORIZONTAL)
    max_speed_scale.set(2)
    max_speed_scale.pack()

    max_force_label = tk.Label(control_frame, text="Max Force")
    max_force_label.pack()
    max_force_scale = tk.Scale(control_frame, from_=0, to=1, resolution=0.01, orient=tk.HORIZONTAL)
    max_force_scale.set(0.05)
    max_force_scale.pack()

    perception_label = tk.Label(control_frame, text="Perception Radius")
    perception_label.pack()
    perception_scale = tk.Scale(control_frame, from_=0, to=200, resolution=1, orient=tk.HORIZONTAL)
    perception_scale.set(50)
    perception_scale.pack()

    align_label = tk.Label(control_frame, text="Alignment Weight")
    align_label.pack()
    align_scale = tk.Scale(control_frame, from_=0, to=5, resolution=0.1, orient=tk.HORIZONTAL)
    align_scale.set(1)
    align_scale.pack()

    cohesion_label = tk.Label(control_frame, text="Cohesion Weight")
    cohesion_label.pack()
    cohesion_scale = tk.Scale(control_frame, from_=0, to=5, resolution=0.1, orient=tk.HORIZONTAL)
    cohesion_scale.set(1)
    cohesion_scale.pack()

    separation_label = tk.Label(control_frame, text="Separation Weight")
    separation_label.pack()
    separation_scale = tk.Scale(control_frame, from_=0, to=5, resolution=0.1, orient=tk.HORIZONTAL)
    separation_scale.set(1)
    separation_scale.pack()

    cluster_label = tk.Label(control_frame, text="Clusters: 0")
    cluster_label.pack()

    def toggle_radius():
        global show_radius
        show_radius = not show_radius

    toggle_button = tk.Button(control_frame, text="Toggle Radius", command=toggle_radius)
    toggle_button.pack()

    # Initialize boids
    boids = []
    for _ in range(100):
        boids.append(Boid(random.uniform(0, WIDTH), random.uniform(0, HEIGHT), WIDTH, HEIGHT))

    # Randomly select a single boid to highlight perception radius
    highlighted_boid = random.choice(boids)

    # Initialize cluster counts and time steps
    cluster_counts = []
    time_steps = []

    # Set up matplotlib figure for cluster count over time
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Clusters')
    ax.set_title('Clusters Over Time')
    line, = ax.plot([], [])

    # Create a FigureCanvasTkAgg object for the graph in the graph frame
    canvas_fig = FigureCanvasTkAgg(fig, master=graph_frame)
    canvas_fig.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def draw_boid(boid):
        x = boid.position.x
        y = boid.position.y
        angle = boid.velocity.heading()
        size = 6
        p1 = (x + size * math.cos(angle), y + size * math.sin(angle))
        p2 = (x + size * math.cos(angle + 2.5), y + size * math.sin(angle + 2.5))
        p3 = (x + size * math.cos(angle - 2.5), y + size * math.sin(angle - 2.5))
        canvas.create_polygon(p1, p2, p3, fill="black")

    def compute_clusters(boids, eps=20, min_samples=2):
        positions = np.array([[boid.position.x, boid.position.y] for boid in boids])
        if len(positions) == 0:
            return 0
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(positions)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        return n_clusters

    def update_boids():
        # Update parameters from sliders
        max_speed = max_speed_scale.get()
        max_force = max_force_scale.get()
        perception_radius = perception_scale.get()
        align_weight = align_scale.get()
        cohesion_weight = cohesion_scale.get()
        separation_weight = separation_scale.get()

        # Update boids
        canvas.delete("all")
        for boid in boids:
            boid.max_speed = max_speed
            boid.max_force = max_force
            boid.perception_radius = perception_radius
            boid.edges()
            boid.flock(boids, align_weight, cohesion_weight, separation_weight)
            boid.update()
            draw_boid(boid)
        
        # Conditionally draw the perception radius for the highlighted boid
        if highlighted_boid and show_radius:
            x = highlighted_boid.position.x
            y = highlighted_boid.position.y
            radius = highlighted_boid.perception_radius
            canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                outline="blue", width=1
            )

        # Compute clusters
        n_clusters = compute_clusters(boids)
        cluster_counts.append(n_clusters)
        time_steps.append(len(cluster_counts))

        # Update cluster label
        cluster_label.config(text=f"Clusters: {n_clusters}")

        # Update plot
        line.set_data(time_steps, cluster_counts)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()

        root.after(10, update_boids)

    update_boids()
    root.mainloop()

if __name__ == "__main__":
    main()
