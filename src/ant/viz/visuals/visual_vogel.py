# src/visuals/visual_vogel.py
# pylint: disable=no-member
import math
import py5
from .visual_base import VisualBase

class VisualVogel(VisualBase):
    """
    VisualVogel creates a dynamic visualization inspired by Vogel's spiral, 
    a mathematical pattern where points are arranged using the golden angle.
    The visualization responds to alpha wave signals, modulating parameters 
    like size, speed, and color to reflect real-time neurofeedback.
    """

    def __init__(self, signal_handler, config):
        """
        Initializes the visual with a signal handler for real-time data input
        and a configuration dictionary for customizable parameters.

        Args:
            signal_handler (SignalHandler): Manages the alpha wave signals.
            config (dict): Contains visual and color settings for the sketch.
        """
        super().__init__(signal_handler, config)
        self.signal_handler = signal_handler
        self.config = config
        self.renderer = py5.P2D

        # Scale and speed factors for visual response to alpha signals
        self.alpha_scale_min = 10  # Minimum scaling for visualization
        self.alpha_scale_max = 30  # Maximum scaling for visualization
        self.alpha_speed_min = 0.01  # Minimum speed for animations
        self.alpha_speed_max = 0.08  # Maximum speed for animations

        # Expected alpha wave signal range and smoothing parameters
        self.alpha_signal_min = 8.0  # Lower bound for alpha wave signal
        self.alpha_signal_max = 12.9  # Upper bound for alpha wave signal
        self.previous_alpha_signal = None  # For storing the previous signal value
        self.smooth_factor = 0.01  # Controls the interpolation rate for signal smoothing
        self.initialized = False  # Tracks if the smoothing has been initialized

        # Number of points in the spiral and overall scaling factor
        self.n = 150  # Number of points in the Vogel's spiral
        self.myscale = 0  # Scaling factor adjusted by the alpha signal

    def draw(self):
        """
        Continuously renders the dynamic Vogel's spiral visualization:
        - Smoothly adjusts visual properties (scale, speed, colors) based on the alpha wave signal.
        - Points are arranged according to the golden angle, creating a natural, flower-like pattern.
        - Uses signal-driven parameters to modulate size and animation speed.
        """
        py5.no_stroke()  # Disable outlines for rendered shapes
        py5.color_mode(py5.HSB, 360, 1.0, 1.0)  # Use HSB color mode for more intuitive color control

        # Retrieve the current alpha wave signal
        current_alpha_signal = self.signal_handler.get_signal()

        if not self.initialized:  # Initialize the smoothing process on the first frame
            self.previous_alpha_signal = current_alpha_signal
            self.initialized = True

        # Smoothly interpolate between the previous and current alpha signal
        alpha_signal = py5.lerp(self.previous_alpha_signal, current_alpha_signal, self.smooth_factor)
        self.previous_alpha_signal = alpha_signal  # Update the previous signal value

        # Map the alpha signal to visual scaling and animation speed
        alpha_scale = py5.remap(alpha_signal, self.alpha_signal_min, self.alpha_signal_max, self.alpha_scale_min, self.alpha_scale_max)
        alpha_speed = py5.remap(alpha_signal, self.alpha_signal_min, self.alpha_signal_max, self.alpha_speed_min, self.alpha_speed_max)
        self.myscale = py5.width / alpha_scale if py5.width < py5.height else py5.height / alpha_scale

        # Set the background color dynamically
        hue = 255 - 60  # Base hue for the background
        saturation = 1.0 - 2.0 * 1.0  # Minimum saturation level
        brightness = 0.5 + 1.0  # Maximum brightness level
        max_color = (hue, hue * saturation, hue * brightness)  # Convert to RGB-like values
        py5.background(*max_color)
        py5.push_matrix()
        py5.no_fill()
        py5.no_stroke()  # Ensure no outlines are drawn for shapes

        # Draw each point in the Vogel's spiral
        for i in range(self.n):
            theta = 2.39996 * i  # Golden angle in radians
            r = self.myscale * math.sqrt(i)  # Radial distance from the center

            # Calculate normalized parameters for dynamic colors
            s = i / self.n  # Normalized position along the spiral
            hue = 255 - 60 * s  # Hue decreases with distance
            saturation = 1.0 - 2.0 * s  # Saturation decreases with distance
            brightness = 0.5 + s  # Brightness increases with distance
            py5.fill(hue, saturation, brightness)  # Set fill color for the point

            # Modulate the size of each point dynamically using a sine wave
            d = self.myscale * (5 + 0.5 * math.sin(py5.frame_count * alpha_speed + r))

            # Calculate the (x, y) position of the point
            x = 0.5 * py5.width + r * math.sin(theta)
            y = 0.5 * py5.height + r * math.cos(theta)

            # Draw the point as an ellipse
            py5.ellipse(x, y, d, d)
        py5.pop_matrix()  # Restore transformation matrix
        super().draw()    
