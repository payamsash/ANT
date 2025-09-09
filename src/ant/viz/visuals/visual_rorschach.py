# src/visuals/visual_rorschach.py
# pylint: disable=no-member
import os
import py5
import py5.surface
from .visual_base import VisualBase

class VisualRorschach(VisualBase):
    """
    Rorschach is a visual generation class that produces a neurofeedback-based visual
    display. It utilizes a fragment shader to render dynamic graphics responding to
    real-time alpha brainwave signals.

    Attributes:
        shader (py5.Shader): Fragment shader that controls the visual rendering.
        signal_handler (SignalHandler): External object that provides real-time brainwave signal data.
        config (dict): Configuration dictionary containing color settings and visual parameters.
        previous_alpha_signal (float): Stores the alpha signal value from the previous frame for interpolation.
        initialized (bool): Indicates if the initial alpha signal has been set.
        smooth_factor (float): Interpolation factor for gradual transitions of the alpha signal.
    """
    
    def __init__(self, signal_handler, config):
        """
        Initializes the Rorschach class with signal handling and configuration data.

        Args:
            signal_handler (SignalHandler): An object providing real-time brainwave signal data.
            config (dict): Configuration settings, including colors and visual parameters.
        """
        super().__init__(signal_handler, config)
        self.shader = None
        self.signal_handler = signal_handler  # Store the signal handler instance
        self.config = config
        self.previous_alpha_signal = None  # Store the alpha signal value from the previous frame
        self.initialized = False  # Flag to set initial signal value only once
        self.smooth_factor = 0.01  # Controls the interpolation speed for smooth transitions
        # Load the fragment shader from the specified file path

    def draw(self):
        """
        Main drawing function that is called every frame.
        It interpolates the alpha signal smoothly and passes the updated signal,
        along with color and time data, to the shader for rendering.
        """
        # Retrieve the current alpha brainwave signal from the signal handler
        current_alpha_signal = self.signal_handler.get_signal()

        # Initialize the previous alpha signal on the first frame
        if not self.initialized:
            self.previous_alpha_signal = current_alpha_signal
            py5.fill(255)
            py5.stroke_weight(1.0)
            py5.no_stroke()
            current_dir = os.path.dirname(__file__)
            shader_path = os.path.join(current_dir, "shaders", "rorschach01.frag")
            self.shader = py5.load_shader(shader_path)
            self.initialized = True

        # Interpolate between the previous and current signals for smooth visual transitions
        alpha_signal = py5.lerp(self.previous_alpha_signal, current_alpha_signal, self.smooth_factor)
        self.previous_alpha_signal = alpha_signal  # Update the previous signal for the next frame
        # Set shader parameters for dynamic visual effects
        self.shader.set("u_resolution", float(py5.width), float(py5.height))  # Screen resolution
        self.shader.set("u_time", py5.millis() / 5000.0)  # Time parameter for animations
        self.shader.set("u_alpha_signal", alpha_signal)  # Pass interpolated alpha signal to shader

        # Load color configurations from the provided settings
        color1 = self.get_color("color1b")
        color2 = self.get_color("color1c")
        background_color = self.get_backgroundcolor("backgroundColor1")

        # Normalize RGB color values (0-255 range to 0-1 range) and set in the shader
        self.shader.set("u_inkColor1", *(c / 255.0 for c in color1))
        self.shader.set("u_inkColor2", *(c / 255.0 for c in color2))
        self.shader.set("u_backgroundColor", *(c / 255.0 for c in background_color))

        # Apply the shader and draw a rectangle covering the entire canvas
        py5.background(*background_color)
        py5.shader(self.shader)
        py5.rect(0, 0, py5.width, py5.height)  # Draw the fullscreen rectangle
        py5.reset_shader()
        super().draw()