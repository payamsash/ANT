# src/visuals/visual_preset6.py
# pylint: disable=no-member
import os
import py5
import py5.surface
from .visual_base import VisualBase

class VisualPreset6(VisualBase):
    """
    VisualPreset6 is a visual generation class that produces a neurofeedback-based visual
    display. It uses a fragment shader to render graphics that respond to real-time alpha
    brainwave signals.

    Attributes:
        shader (py5.Shader): Fragment shader that controls the display.
        signal_handler (SignalHandler): An external object responsible for retrieving brainwave signals.
        config (dict): Configuration dictionary with color and other rendering settings.
        previous_alpha_signal (float): Stores the alpha signal value from the previous frame.
        initialized (bool): Tracks if the initial alpha signal has been set.
        smooth_factor (float): Interpolation factor for smooth signal transitions.
    """
    def __init__(self, signal_handler, config):
        """
        Initializes the VisualPreset6 class with signal handling and configuration data.

        Args:
            signal_handler (SignalHandler): Object that provides real-time signal data.
            config (dict): Configuration settings, including colors and visual parameters.
        """
        super().__init__(signal_handler, config)
        self.shader = None
        self.signal_handler = signal_handler  # Store the signal handler instance
        self.config = config
        self.previous_alpha_signal = None  # Previous frame's alpha signal value
        self.initialized = False  # Flag to set initial signal only once
        self.smooth_factor = 0.005  # Factor controlling signal smoothing speed

    def settings(self):
        """
        Configures the initial display settings, such as window size and icon.
        """
        if self.is_full_screen:
            py5.full_screen(py5.P2D)  # Full screen mode with P2D renderer
        else:
            py5.size(800, 800, py5.P2D)  # Windowed mode with P2D renderer
            # Set custom icon for the display window
            PJOGL = py5.JClass("processing.opengl.PJOGL")
            PJOGL.setIcon("assets/images/epflecal_lab-logo.png")

    def setup(self):
        """
        Sets up the shader and display properties. Loads the shader file and configures
        the window's appearance.
        """
        py5.no_stroke()  # Disable strokes for clean visuals
        # Load the fragment shader
        current_dir = os.path.dirname(__file__)
        shader_path = os.path.join(current_dir, "shaders", "rosace.frag")
        self.shader = py5.load_shader(shader_path)
        
        # Configure display window properties
        surface = py5.get_surface()
        surface.set_always_on_top(True)
        surface.set_location(0, 0)
        surface.set_title("Advancing Neurofeedback in Tinnitus - Visual Preset 6")

    def draw(self):
        """
        Main drawing function that updates each frame. Interpolates the alpha signal
        smoothly and passes it along with color and time data to the shader for rendering.
        """
        # Retrieve the current signal from the signal handler
        current_alpha_signal = self.signal_handler.get_signal()
        
        # Initialize the previous signal on the first frame
        if not self.initialized:
            self.previous_alpha_signal = current_alpha_signal
            self.initialized = True
        
        # Interpolate between previous and current signals for smooth transitions
        alpha_signal = py5.lerp(self.previous_alpha_signal, current_alpha_signal, self.smooth_factor)
        self.previous_alpha_signal = alpha_signal  # Update previous signal for next frame

        # Set shader parameters for visual effects
        self.shader.set("u_resolution", float(py5.width), float(py5.height))  # Set screen resolution
        self.shader.set("u_time", py5.millis() / 1000)  # Time uniform for animation speed
        self.shader.set("u_alpha_signal", alpha_signal)  # Pass interpolated alpha signal to shader

        # Load colors from the configuration
        inkColor1 = self.config['colors']['inkColor1']
        paperColor = self.config['colors']['paperColor']

        # Pass color settings to the shader, normalized to 0-1 range
        self.shader.set("u_inkColor1", inkColor1[0] / 255.0, inkColor1[1] / 255.0, inkColor1[2] / 255.0)
        self.shader.set("u_paperColor", paperColor[0] / 255.0, paperColor[1] / 255.0, paperColor[2] / 255.0)

        # Apply shader and draw the rectangle covering the entire canvas
        py5.shader(self.shader)
        py5.rect(0, 0, py5.width, py5.height)
