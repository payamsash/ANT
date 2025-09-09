# src/visuals/visual_galaxy_spiral.py
# pylint: disable=no-member
import os
import py5
import py5.surface
from .visual_base import VisualBase

class VisualGalaxySpiral(VisualBase):
    """
    VisualGalaxySpiral is a visual generation class that creates a neurofeedback-based visual
    display using a fragment shader. The visual responds to real-time alpha brainwave signals,
    creating a dynamic spiral galaxy effect.

    Attributes:
        shader (py5.Shader): Fragment shader used for rendering the visual effects.
        signal_handler (SignalHandler): An object that provides real-time alpha brainwave signals.
        config (dict): Configuration dictionary containing colors and other visual parameters.
        previous_alpha_signal (float): Stores the alpha signal value from the previous frame
                                       to enable smooth transitions.
        initialized (bool): Flag to check if the initial alpha signal value has been set.
        smooth_factor (float): Interpolation factor controlling the speed of signal smoothing.
    """

    def __init__(self, signal_handler, config):
        """
        Initializes the VisualGalaxySpiral class with a signal handler and configuration settings.

        Args:
            signal_handler (SignalHandler): An object for retrieving real-time alpha brainwave signals.
            config (dict): Configuration settings, including color definitions and visual parameters.
        """
        super().__init__(signal_handler, config)
        self.shader = None  # Placeholder for the fragment shader
        self.signal_handler = signal_handler  # Store the signal handler instance
        self.config = config  # Store the configuration dictionary
        self.previous_alpha_signal = None  # Alpha signal from the previous frame
        self.initialized = False  # Flag to ensure initial signal setup only once
        self.smooth_factor = 0.01  # Controls the interpolation speed of alpha signal smoothing

    def draw(self):
        """
        Main drawing function that updates every frame.
        It interpolates the alpha signal smoothly and passes it, along with time,
        color, and other parameters, to the shader for rendering the visual effects.
        """
        # Retrieve the current alpha brainwave signal from the signal handler
        current_alpha_signal = self.signal_handler.get_signal()

        # Initialize the previous alpha signal on the first frame to avoid sudden jumps
        if not self.initialized:
            self.previous_alpha_signal = current_alpha_signal
            py5.fill(255)
            py5.stroke_weight(1.0)
            py5.no_stroke()
            # Load the fragment shader from the specified path
            current_dir = os.path.dirname(__file__)
            shader_path = os.path.join(current_dir, "shaders", "galaxy_spiral.frag")
            self.shader = py5.load_shader(shader_path)
            self.initialized = True

        # Interpolate between the previous and current signals for smooth visual transitions
        alpha_signal = py5.lerp(self.previous_alpha_signal, current_alpha_signal, self.smooth_factor)
        self.previous_alpha_signal = alpha_signal  # Update the previous signal for the next frame

        # Set shader parameters for rendering the visual effects
        self.shader.set("u_resolution", float(py5.width), float(py5.height))  # Set screen resolution
        self.shader.set("u_time", py5.millis() / 1000.0)  # Set normalized time parameter for animations
        self.shader.set("u_alpha_signal", alpha_signal)  # Pass the smoothed alpha signal to the shader

        # Retrieve color configurations from the config dictionary
        primary_color = self.get_color("color3a")
        background_color = self.get_backgroundcolor("backgroundColor3")

        # Normalize RGB values to the 0-1 range and pass them to the shader
        self.shader.set("u_color", *(c / 255.0 for c in primary_color))
        self.shader.set("u_backgroundColor", *(c / 255.0 for c in background_color))

        # Apply the shader and draw a rectangle that covers the entire canvas
        py5.background(*background_color)
        py5.shader(self.shader)
        py5.rect(0, 0, py5.width, py5.height)  # Draw the rectangle with the shader effect
        py5.reset_shader()
        super().draw()