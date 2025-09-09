# src/visuals/visual_rings.py
# pylint: disable=no-member
import py5
from .visual_base import VisualBase

class VisualRings(VisualBase):
    """
    VisualRings is a dynamic visual effect responsive to real-time alpha brainwave signals.
    It generates concentric rings that adjust their size, speed, and noise-based deformation
    based on the alpha signal input, creating a fluid and evolving pattern.

    Attributes:
        signal_handler (SignalHandler): Provides real-time alpha brainwave signals.
        config (dict): Stores configuration settings such as color schemes.
        alpha_signal_min (float): Minimum expected alpha signal value (8 Hz).
        alpha_signal_max (float): Maximum expected alpha signal value (13 Hz).
        alpha_scale_min (float): Minimum scaling factor for visual elements.
        alpha_scale_max (float): Maximum scaling factor for visual elements.
        alpha_speed_min (float): Minimum speed for time-based animation.
        alpha_speed_max (float): Maximum speed for time-based animation.
        alpha_amp_min (float): Minimum amplitude for ring deformation.
        alpha_amp_max (float): Maximum amplitude for ring deformation.
        alpha_res_min (int): Minimum resolution for the ring's shape (number of vertices).
        alpha_res_max (int): Maximum resolution for the ring's shape (number of vertices).
        previous_alpha_signal (float): Stores the last alpha signal for smoothing.
        initialized (bool): Flag to indicate if the initial alpha signal is set.
        smooth_factor (float): Interpolation factor for smoothing alpha signal changes.
        skip_circles (int): Number of inner circles to skip drawing.
        speed (float): Multiplier for the time progression in the noise function.
        alpha_iter (int): Number of concentric rings (layers) to draw.
        time (float): Time variable for animating motion.
    """

    def __init__(self, signal_handler, config):
        """
        Initializes the VisualRings class with a signal handler and configuration settings.

        Args:
            signal_handler (SignalHandler): Object for retrieving real-time alpha signals.
            config (dict): Dictionary containing color settings and other visual parameters.
        """
        super().__init__(signal_handler, config)
        self.signal_handler = signal_handler
        self.config = config

        # Alpha signal parameters for smooth scaling and visual dynamics
        self.alpha_signal_min = 8
        self.alpha_signal_max = 13
        self.alpha_scale_min = 1.5
        self.alpha_scale_max = 5.0
        self.alpha_speed_min = 0.4
        self.alpha_speed_max = 1.0
        self.alpha_amp_min = 5
        self.alpha_amp_max = 15
        self.alpha_res_min = 15
        self.alpha_res_max = 60

        # State management for alpha signal and initialization
        self.previous_alpha_signal = None
        self.initialized = False
        self.smooth_factor = 0.01
        self.skip_circles = 5

        # Parameters for dynamic noise-based ring shapes
        self.speed = 0.0025
        self.alpha_iter = 30
        self.time = 0
        self.color = self.get_color("color7")
        self.background_color = self.get_backgroundcolor("backgroundColor7")

    def draw(self):
        """
        Main drawing function executed every frame.
        It processes the alpha signal, maps it to visual parameters, and draws concentric
        noise-deformed rings that evolve over time based on the alpha brainwave input.
        """
        # Retrieve the latest alpha brainwave signal
        current_alpha_signal = self.signal_handler.get_signal()

        # Initialize the previous alpha signal to prevent abrupt changes
        if not self.initialized:
            py5.no_fill()  # Disable shape filling for outlined rings
            py5.stroke_weight(1.0)
            self.previous_alpha_signal = current_alpha_signal
            self.initialized = True

        # Smooth the transition between the previous and current alpha signals
        alpha_signal = py5.lerp(self.previous_alpha_signal, current_alpha_signal, self.smooth_factor)
        self.previous_alpha_signal = alpha_signal  # Update for next frame

        # Map the alpha signal to various visual properties
        alpha_scale = py5.remap(alpha_signal, self.alpha_signal_min, self.alpha_signal_max,
                                self.alpha_scale_max, self.alpha_scale_min)
        alpha_speed = py5.remap(alpha_signal, self.alpha_signal_min, self.alpha_signal_max,
                                self.alpha_speed_min, self.alpha_speed_max)

        # Map the signal to dynamic amplitude and resolution
        dynamic_amp = py5.remap(alpha_signal, self.alpha_signal_min, self.alpha_signal_max,
                                self.alpha_amp_max, self.alpha_amp_min)
        dynamic_res = int(py5.remap(alpha_signal, self.alpha_signal_min, self.alpha_signal_max,
                                    self.alpha_res_max, self.alpha_res_min))

        py5.push_matrix()
        py5.no_fill()
        py5.background(*self.background_color)

        # Set the stroke (line color) from the configuration
        py5.stroke(*self.color)

        # Center the drawing on the canvas
        py5.translate(py5.width / 2, py5.height / 2)

        # Update the time variable for continuous motion
        self.time += self.speed * alpha_speed
        # Calculate offsets for noise generation
        x_offset = py5.sin(self.time * 0.05) * 100 + 500
        y_offset = py5.cos(self.time * 0.05) * 100 + 500

        # Draw concentric rings with noise deformation
        for j in range(self.skip_circles, self.alpha_iter):
            radius = j * dynamic_amp * alpha_scale  # Precompute radius
            stroke_weight = j / self.alpha_iter * 2.5
            py5.stroke_weight(stroke_weight)
            py5.begin_shape()
            for i in range(dynamic_res):  # Iterate over vertices using dynamic resolution
                angle = py5.TWO_PI * i / dynamic_res  # Calculate angle for vertex placement

                # Calculate noise-based deformation
                noise_val = py5.noise(
                    x_offset + py5.sin(angle) * 0.015 * j,
                    y_offset + py5.cos(angle) * 0.015 * j,
                    self.time
                )
                # Determine vertex position with noise modulation
                x = py5.sin(angle) * radius * noise_val
                y = py5.cos(angle) * radius * noise_val
                py5.vertex(x, y)
            py5.end_shape(py5.CLOSE)  # Complete the shape and close the ring
        py5.pop_matrix()
        super().draw()