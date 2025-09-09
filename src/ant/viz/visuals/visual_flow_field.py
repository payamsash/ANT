# src/visuals/visual_flow_field.py
# pylint: disable=no-member
import random
from math import cos, sin, pi
import py5
from .visual_base import VisualBase

class VisualFlowField(VisualBase):
    """
    VisualFlowField generates a dynamic visual based on an alpha signal.
    Particles move within a circular area, and their behavior is influenced 
    by the alpha signal input.
    """

    def __init__(self, signal_handler, config):
        """
        Initialize the visual preset with signal handler and configuration.

        Parameters:
        - signal_handler: Object for receiving alpha signals.
        - config: Configuration dictionary for visual properties like colors.
        """
        super().__init__(signal_handler, config)
        self.signal_handler = signal_handler
        self.config = config
        
        # Particle setup
        self.particles = []
        self.nums = 150  # Total number of particles

        # Alpha signal parameters
        self.alpha_signal_min = 8
        self.alpha_signal_max = 13
        self.alpha_scale_min = 25
        self.alpha_scale_max = 450
        self.alpha_particles_min = 5
        self.alpha_particles_max = self.nums
        self.previous_alpha_signal = None
        self.initialized = False
        self.smooth_factor = 0.02  # Increased for better responsiveness
        self.color = self.get_color("color3a")
        self.background_color = self.get_backgroundcolor("backgroundColor3")

    def draw(self):
        """Draw the dynamic visual responding to the alpha signal."""

        current_alpha_signal = self.signal_handler.get_signal()

        # Initialize the previous alpha signal on the first frame
        if not self.initialized:
            self.previous_alpha_signal = current_alpha_signal
            py5.fill(255)
            py5.background(*self.background_color)
            py5.no_stroke()
            py5.stroke_weight(1.0)
            self.init_particles()
            self.initialized = True

        # Smoothly interpolate the alpha signal
        alpha_signal = py5.lerp(self.previous_alpha_signal, current_alpha_signal, self.smooth_factor)
        self.previous_alpha_signal = alpha_signal

        # Map the alpha signal to visual parameters
        alpha_scale = py5.remap(alpha_signal, self.alpha_signal_min, self.alpha_signal_max, self.alpha_scale_max, self.alpha_scale_min)
        alpha_particles_num = int(py5.remap(alpha_signal, self.alpha_signal_min, self.alpha_signal_max, self.alpha_particles_max, self.alpha_particles_min))
        py5.background(*self.background_color)
        py5.stroke(*self.color)
        # Update and display particles
        for i in range(alpha_particles_num):
            radius = py5.remap(i, 0, self.nums, 3, 6)  # Remap particle size based on index
            particle = self.particles[i]
            particle.move()
            particle.check_edge(alpha_scale)
            particle.display(radius, self.color, alpha_scale)

        # Display remaining particles as static points
        for i in range(alpha_particles_num, self.nums):
            self.particles[i].display(1, self.color, alpha_scale, 0)

        super().draw()


    def init_particles(self):
        """Initialize particles within a circular boundary."""
        center_x, center_y = py5.width / 2, py5.height / 2
        for _ in range(self.nums):
            angle = random.uniform(0, 2 * pi)
            radius = random.uniform(0, self.alpha_scale_max)
            x = center_x + radius * cos(angle)
            y = center_y + radius * sin(angle)
            direction = py5.Py5Vector(cos(random.uniform(0, 2 * pi)), sin(random.uniform(0, 2 * pi)))
            self.particles.append(self.Particle(x, y, direction))

    class Particle:
        """Class representing a single particle in the visual."""

        def __init__(self, x, y, direction):
            """
            Initialize the particle's position and movement direction.

            Parameters:
            - x: Initial x-coordinate of the particle.
            - y: Initial y-coordinate of the particle.
            - direction: Initial movement direction vector.
            """
            self.pos = py5.Py5Vector(x, y)  # Current position
            self.dir = direction
            self.history = []
            self.history_length = random.uniform(10, 50)
            self.noise_scale = 200  # Lowered for finer control
            self.fading = False
            self.fade_progress = 0

        def move(self):
            """Move the particle based on Perlin noise and a vector toward the center."""
            center_x, center_y = py5.width / 2, py5.height / 2
            noise_angle = py5.noise(
                self.pos.x / self.noise_scale, 
                self.pos.y / self.noise_scale, 
                py5.frame_count / self.noise_scale
            ) * 2 * pi  # Angle from Perlin noise
            
            # Create a vector pointing toward the center
            center_vector = py5.Py5Vector(center_x, center_y) - self.pos
            center_vector.normalize()  # Normalize the vector
            
            # Direction influenced by noise and center attraction
            self.dir.x = cos(noise_angle) + sin(noise_angle) - sin(noise_angle) + center_vector.x
            self.dir.y = sin(noise_angle) - cos(noise_angle) * sin(noise_angle) + center_vector.y
            self.dir.normalize()  # Normalize final direction
            
            velocity = self.dir.copy * 1.1  # Adjust speed if needed
            self.pos += velocity  # Update position

            # Store position history for the trail effect
            self.history.append(self.pos.copy)
            if len(self.history) > self.history_length:
                self.history.pop(0)

        def check_edge(self, alpha_scale):
            """Check if the particle is outside the alpha_scale radius boundary and reposition if necessary."""
            center_x, center_y = py5.width / 2, py5.height / 2
            center_vector = py5.Py5Vector(center_x, center_y) - self.pos
            center_vector.normalize()
            alignment = self.dir.dot(center_vector)  # Cosine of angle between vectors
            distance_squared = (self.pos.x - center_x) ** 2 + (self.pos.y - center_y) ** 2
            if distance_squared >= alpha_scale ** 2:
                angle = random.uniform(0, 2 * pi)
                self.pos.x = center_x + alpha_scale * cos(angle)
                self.pos.y = center_y + alpha_scale * sin(angle)
                self.history.clear()
            if alignment < 0.1:
                self.fading = True

        # def check_edge(self, alpha_scale):
        #     """Check if particle is outside the alpha_scale radius boundary and reposition if necessary."""
        #     center_x, center_y = py5.width / 2, py5.height / 2  # Center coordinates
        #     if (pow(self.pos.x - center_x, 2) + pow(self.pos.y - center_y, 2)) >= pow(alpha_scale, 2):
        #         # If outside, reposition randomly within the window bounds
        #         self.pos.x = random.uniform(0, py5.width)
        #         self.pos.y = random.uniform(0, py5.height)

        def display(self, radius, color, alpha_scale, alpha=255):
            """Display the particle and its trail."""
            if self.fading:
                # Reduce alpha values in the trail
                for i, past_pos in enumerate(self.history):
                    trail_alpha = py5.lerp(0, alpha, i / len(self.history)) * (1 - self.fade_progress)
                    py5.fill(*color, trail_alpha)
                    py5.ellipse(past_pos.x, past_pos.y, radius, radius)
                # Increment fade progress
                self.fade_progress += 0.05  # Adjust speed as needed
                if self.fade_progress >= 1.0:
                    self.fading = False  # Reset fading state
                    self.fade_progress = 0
                    self.history.clear()  # Clear history after fade-out
                    center_x, center_y = py5.width / 2, py5.height / 2
                    self.pos.x = center_x + alpha_scale
                    self.pos.y = center_y + alpha_scale
            else:
                # Normal display
                for i, past_pos in enumerate(self.history):
                    trail_alpha = py5.lerp(0, alpha, i / len(self.history))
                    py5.fill(*color, trail_alpha)
                    py5.ellipse(past_pos.x, past_pos.y, radius, radius)
            # Draw current position
            # py5.fill(*color, alpha)  # Set color with full alpha
            # py5.ellipse(self.pos.x, self.pos.y, radius, radius)  # Draw current position as ellipse


