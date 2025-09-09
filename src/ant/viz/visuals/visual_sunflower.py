# src/visuals/visual_sunflower.py
# pylint: disable=no-member
import os
import py5
import py5.surface
from .visual_base import VisualBase

class VisualSunflower(VisualBase):
    """
    Sunflower is a visual generation class that creates a neurofeedback-based visual display.
    It uses a fragment shader to render dynamic graphics in response to real-time alpha
    brainwave signals.

    Attributes:
        shader (py5.Shader): Fragment shader for rendering the visual effects.
        signal_handler (SignalHandler): An object that provides real-time alpha brainwave signals.
        config (dict): Configuration dictionary containing color settings and other visual parameters.
        previous_alpha_signal (float): Stores the previous alpha signal for smooth interpolation.
        initialized (bool): Flag to check if the initial alpha signal has been set.
        smooth_factor (float): Interpolation factor to control the smoothing speed of signal transitions.
    """
    
    def __init__(self, signal_handler, config):
        """
        Initializes the Sunflower visual with signal handling and configuration settings.

        Args:
            signal_handler (SignalHandler): An object to retrieve real-time alpha brainwave signals.
            config (dict): Configuration settings, including color definitions and visual parameters.
        """
        super().__init__(signal_handler, config)
        self.shader = None
        self.signal_handler = signal_handler
        self.config = config
        self.previous_alpha_signal = None  # Store alpha signal from the previous frame
        self.initialized = False  # Ensure initial signal setup is only done once
        self.smooth_factor = 0.01  # Controls interpolation speed for smooth signal transition

    def draw(self):
        """
        Main drawing function that updates each frame.
        It interpolates the alpha signal smoothly and passes it, along with time,
        color, and speed factor data, to the shader for rendering the visual effects.
        """
        # Retrieve the current alpha brainwave signal from the signal handler
        current_alpha_signal = self.signal_handler.get_signal()

        # Initialize the previous alpha signal on the first frame
        if not self.initialized:
            self.previous_alpha_signal = current_alpha_signal
            py5.fill(255)
            py5.stroke_weight(1.0)
            py5.no_stroke()
            # Load the fragment shader from the specified path
            current_dir = os.path.dirname(__file__)
            shader_path = os.path.join(current_dir, "shaders", "sunflower.frag")
            self.shader = py5.load_shader(shader_path)
            self.initialized = True

        # Interpolate between previous and current signals for smooth transitions
        alpha_signal = py5.lerp(self.previous_alpha_signal, current_alpha_signal, self.smooth_factor)
        self.previous_alpha_signal = alpha_signal  # Update previous signal for the next frame

        # Set shader parameters for rendering the visual effects
        self.shader.set("u_resolution", float(py5.width), float(py5.height))  # Screen resolution
        self.shader.set("u_time", py5.millis() / 5000.0)  # Normalized time parameter for animations
        self.shader.set("u_alpha_signal", alpha_signal)  # Pass the smoothed alpha signal to the shader

        # Load color settings from the configuration dictionary
        color = self.get_color("color4a")
        background_color = self.get_backgroundcolor("paperColor")

        # Normalize RGB color values to the 0-1 range and set them in the shader
        self.shader.set("u_color", *(c / 255.0 for c in color))
        self.shader.set("u_backgroundColor", *(c / 255.0 for c in background_color))

        # Apply the shader and draw a rectangle covering the entire canvas
        py5.background(*background_color)
        py5.shader(self.shader)
        py5.rect(0, 0, py5.width, py5.height)  # Draw the fullscreen rectangle
        py5.reset_shader()
        super().draw()