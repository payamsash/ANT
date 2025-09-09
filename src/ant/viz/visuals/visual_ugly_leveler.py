# src/visuals/visual_ugly_leveler.py
# pylint: disable=no-member
import random
import py5
from .visual_base import VisualBase

OK_ALPHA = 9.0
MAX_ALPHA = 13.0

# Global variable to represent signal value
signal_value = 0  # Value starts at 0 and evolves to 100
signal_direction = 1  # Direction of signal evolution (1 for up, -1 for down)

# Variables to track square positions
square1_x, square1_y = 200, 200
square2_x, square2_y = 200, 200

# Speed of square movement when signal is not 100%
movement_speed = 2

# Smooth movement target positions
square1_target_x, square1_target_y = 0, 0
square2_target_x, square2_target_y = 0, 0

# Interpolation factor for smooth movement
lerp_factor = 0.1

# Bounds for movement around the center
center_bound = 100

class VisualUglyLeveler(VisualBase):
    """
    This class creates a neurofeedback-based visualization where two squares dynamically
    move and shift based on real-time alpha brainwave signals.
    """

    def __init__(self, signal_handler, config):
        """
        Initializes the VisualUglyLeveler.
        
        Parameters:
        signal_handler : object
            Handles the incoming alpha brainwave signals.
        config : dict
            Configuration parameters for visualization settings.
        """
        super().__init__(signal_handler, config)
        self.signal_handler = signal_handler
        self.config = config
        self.initialized = False  # Flag to ensure initial signal setup only once
        self.smooth_factor = 0.01
        self.renderer = py5.P2D
        self.previous_alpha_signal = None  # Alpha signal from the previous frame

    def draw(self):
        """
        Main drawing function executed every frame.
        It processes the alpha signal, maps it to visual parameters, and moves squares
        dynamically based on alpha brainwave input.
        """
        # Retrieve the latest alpha brainwave signal
        current_alpha_signal = self.signal_handler.get_signal()

        # Initialize the previous alpha signal to prevent abrupt changes
        if not self.initialized:
            self.previous_alpha_signal = current_alpha_signal
            py5.no_stroke()
            py5.stroke_weight(1.0)
            py5.no_fill()  # Disable shape filling
            # Initialize positions to the center of the window
            global square1_x, square1_y, square2_x, square2_y
            global square1_target_x, square1_target_y, square2_target_x, square2_target_y
            square1_x = square1_target_x = py5.width / 2
            square1_y = square1_target_y = py5.height / 2
            square2_x = square2_target_x = py5.width / 2
            square2_y = square2_target_y = py5.height / 2
            self.initialized = True

        # Smooth the transition between the previous and current alpha signals
        alpha_signal = py5.lerp(self.previous_alpha_signal, current_alpha_signal, self.smooth_factor)
        self.previous_alpha_signal = alpha_signal  # Update for next frame
        tilt_goal = py5.remap(alpha_signal, OK_ALPHA, MAX_ALPHA, 0, 100)
        #print(tilt_goal)

        global square1_x, square1_y, square2_x, square2_y
        global square1_target_x, square1_target_y, square2_target_x, square2_target_y
        global signal_value, signal_direction

        # Set the background color
        color_background = [229, 229, 229]
        py5.background(*color_background)

        # Center positions based on signal value
        center_x, center_y = py5.width / 2, py5.height / 2

        # If signal is 100%, keep the squares centered
        if signal_value == 100:
            square1_target_x, square1_target_y = center_x, center_y
            square2_target_x, square2_target_y = center_x, center_y
        else:
            # Move target positions randomly within a bounded area
            square1_target_x += random.uniform(-movement_speed, movement_speed)
            square1_target_y += random.uniform(-movement_speed, movement_speed)
            square2_target_x += random.uniform(-movement_speed, movement_speed)
            square2_target_y += random.uniform(-movement_speed, movement_speed)

            # Keep target positions within bounds around the center
            square1_target_x = py5.constrain(square1_target_x, center_x - center_bound, center_x + center_bound)
            square1_target_y = py5.constrain(square1_target_y, center_y - center_bound, center_y + center_bound)
            square2_target_x = py5.constrain(square2_target_x, center_x - center_bound, center_x + center_bound)
            square2_target_y = py5.constrain(square2_target_y, center_y - center_bound, center_y + center_bound)

        # Smoothly move squares towards target positions
        square1_x = py5.lerp(square1_x, square1_target_x, lerp_factor)
        square1_y = py5.lerp(square1_y, square1_target_y, lerp_factor)
        square2_x = py5.lerp(square2_x, square2_target_x, lerp_factor)
        square2_y = py5.lerp(square2_y, square2_target_y, lerp_factor)

        # Draw the first square
        py5.push_matrix()
        py5.translate(square1_x, square1_y)
        py5.rotate(py5.radians(45))
        py5.fill(255)
        py5.stroke(0)
        py5.stroke_weight(4)
        py5.rect(-100, -10, 400, 20)
        py5.pop_matrix()

        # Draw the second square
        py5.push_matrix()
        py5.translate(square2_x, square2_y)
        py5.rotate(py5.radians(-45))
        py5.fill(255)
        py5.stroke(0)
        py5.stroke_weight(1)
        py5.rect(-50, -20, 200, 40, 30)
        py5.pop_matrix()

        super().draw()
