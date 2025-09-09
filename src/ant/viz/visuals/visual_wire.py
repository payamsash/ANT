# src/visuals/visual_wire.py
# pylint: disable=no-member
import random
import math
import py5
from .visual_base import VisualBase

class VisualWire(VisualBase):
    """
    VisualWire generates a dynamic 3D visualization resembling a low-poly sphere. 
    The sphere's geometry is constructed from interconnected vertices, with its 
    structure responding in real-time to alpha wave signals. Key features include:

    - Smooth transitions in sphere growth and shape based on the alpha signal.
    - Dynamic rotation for immersive visual effects.
    - Adjustable geometric properties such as latitudes, resolution, and bending factors.
    - Skipping rendering at the poles for a clean and focused appearance.

    The visualization is designed for neurofeedback applications, adapting its form 
    and motion to reflect changes in the alpha brainwave activity of the user.
    """

    def __init__(self, signal_handler, config):
        """
        Initializes the visual with a signal handler and configuration parameters.

        Args:
            signal_handler (SignalHandler): Object managing input signals.
            config (dict): Configuration for colors and other visual properties.
        """
        super().__init__(signal_handler, config)
        self.renderer = py5.P3D
        self.signal_handler = signal_handler
        self.config = config
        self.background_color = None
        # Sphere geometry parameters
        self.num_latitudes = random.choice([46, 62, 72, 96])  # Determines resolution along latitudes
        self.sphere_radius = 200  # Radius of the sphere
        self.scaling_factor = 6  # Multiplier for enhanced resolution
        self.sphere_res = 360 * self.scaling_factor  # Resolution of the sphere
        self.sphere_range_min = -90 * self.scaling_factor  # Minimum range for sphere geometry
        self.sphere_range_max = 91 * self.scaling_factor  # Maximum range for sphere geometry
        self.skip_at_pole = 6 * self.scaling_factor  # Threshold to skip rendering near poles
        self.bending_factor = random.randint(self.scaling_factor + 1, int(self.scaling_factor * 1.5))  # Controls sphere distortion

        # Alpha wave response parameters
        self.alpha_signal_min = 8.0  # Minimum value of alpha wave signal
        self.alpha_signal_max = 12.9  # Maximum value of alpha wave signal
        self.previous_alpha_signal = None  # Used to smooth transitions
        self.smooth_factor = 0.008  # Controls signal smoothing rate
        self.initialized = False  # Indicates if alpha signal smoothing has been initialized
        self.color = self.get_color("color4a")
        self.background_color = self.get_backgroundcolor('backgroundColor4')

    def draw(self):
        """
        Render the visual by dynamically generating sphere vertices that respond to alpha wave input.
        """
        
        # Retrieve and smooth the alpha wave signal
        current_alpha_signal = self.signal_handler.get_signal()
        if not self.initialized:  # Initialize smoothing on first frame
            self.previous_alpha_signal = current_alpha_signal
            # Load colors from configuration
            py5.stroke_weight(3)  # Set stroke thickness
            self.initialized = True

        alpha_signal = py5.lerp(self.previous_alpha_signal, current_alpha_signal, self.smooth_factor)
        self.previous_alpha_signal = alpha_signal
        py5.stroke(*self.color)  # Set stroke color
        py5.background(*self.background_color)  # Set background color
        # Calculate the growth factor for the sphere based on alpha signal
        growth_factor = int(py5.remap(
            alpha_signal,
            self.alpha_signal_min,
            self.alpha_signal_max,
            self.sphere_range_max,
            self.sphere_range_min + self.skip_at_pole * 2
        ))

        # Reset latitudes and bending factor if alpha signal drops near the minimum threshold
        if alpha_signal <= self.alpha_signal_min + 0.1:
            self.num_latitudes = random.choice([46, 62, 72, 96])
            self.bending_factor = random.randint(self.scaling_factor + 1, int(self.scaling_factor * 1.5))

        # Center the scene and apply dynamic rotation
        py5.push_matrix()
        py5.no_fill()
        py5.translate(py5.width / 2, py5.height / 2)
        py5.rotate_x(py5.frame_count * 0.00125)  # Slow rotation on the X-axis
        py5.rotate_y(py5.frame_count * 0.00075)  # Slow rotation on the Y-axis

        # Generate and render the sphere
        for i in range(0, self.sphere_res, self.num_latitudes * self.scaling_factor):
            theta_a = math.radians(i * 360 / self.sphere_res)  # Calculate angular step in latitude

            py5.begin_shape()
            for j in range(self.sphere_range_min, growth_factor):
                # Skip regions near poles for cleaner visuals
                if j < self.sphere_range_min + self.skip_at_pole or j > self.sphere_range_max - self.skip_at_pole:
                    continue

                # Calculate distortion factor based on bending
                m = -1 * j / (self.scaling_factor / self.bending_factor)
                theta_b = math.radians(j * 360 / self.sphere_res)  # Calculate angular step in longitude

                # Calculate vertex position in 3D space
                x = self.sphere_radius * math.cos(theta_a + math.radians(j + m)) * math.cos(theta_b)
                y = self.sphere_radius * math.sin(theta_a + math.radians(j + m)) * math.cos(theta_b)
                z = self.sphere_radius * math.sin(theta_b)
                py5.vertex(x, y, z)  # Add vertex to the shape
            py5.end_shape()  # Complete the current shape
        py5.pop_matrix()  # Restore transformation matrix
        super().draw()
