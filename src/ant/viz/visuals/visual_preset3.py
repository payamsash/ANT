# src/visuals/visual_preset3.py
# pylint: disable=no-member

import time
import random
import py5
import py5.surface
from .visual_base import VisualBase
from .fall import Fall

class VisualPreset3(VisualBase):
    """
    VisualPreset3 class represents a dynamic L-system tree visualization.
    This class generates and renders a tree-like structure that evolves based on an alpha signal.
    The tree's complexity and growth are influenced by the incoming signal, creating an interactive
    and responsive visual experience.
    """

    def __init__(self, signal_handler, config):
        """Initialize the visual preset with signal handler and configuration."""
        super().__init__(signal_handler, config)
        self.signal_handler = signal_handler
        self.config = config
        # L-system parameters
        self.angle = 25
        self.rules = [
            {"a": "F", "b": "F[+FXF]F[-FXF]"},  # Rule for branches  #"+F[+FX+]-F[X]"
            {"a": "X", "b": "FXF"}  # Rule for leaf nodes #F+F[X]-F[X]
        ]      
        # Visual and animation parameters
        self.smooth_factor = 0.01
        self.start_time = time.time()
        self.rotation_speed = 2.5 * 360 * py5.TWO_PI / 360
        self.distance_branches = 80 
        # Signal and scaling parameters
        self.alpha_signal_min = 8
        self.alpha_signal_max = 13
        self.alpha_scale_min = 1
        self.alpha_scale_max = 4       
        # Additional effects
        self.falls = []
        # Tree structure data
        self.branch_lengths = []
        # State tracking
        self.previous_generations = None
        self.previous_alpha_signal = None  # Initialize to None to be set later
        self.initialized = False  # Flag to set the initial signal only once
        # Precompute L-systems for generations 1 to 4
        self.precomputed_lsystems = {}
        for gen in range(1, 5):
            self.precomputed_lsystems[gen] = self.generate_lsystem(gen)

    def settings(self):
        """Set up the window settings."""
        if self.is_full_screen:
            py5.full_screen(py5.P3D)
        else:
            py5.size(900, 900, py5.P3D)
            PJOGL = py5.JClass("processing.opengl.PJOGL")
            PJOGL.setIcon("assets/images/epflecal_lab-logo.png")

    def setup(self):
        """Set up the initial visual environment."""
        py5.no_stroke()
        py5.frame_rate(20)
        surface = py5.get_surface()
        surface.set_always_on_top(True)
        surface.set_location(0, 0)
        surface.set_title("Advancing Neurofeedback in Tinnitus - Visual Preset 3")

    def draw(self):
        """Main drawing function called every frame."""
        # Set up the 3D view
        py5.perspective(py5.PI/3.0, py5.width/py5.height, 0.1, 1000)
        py5.camera(py5.width/2, py5.height/2, (py5.height/2) / py5.tan(py5.PI/6), 
                   py5.width/2, py5.height/2, 0, 
                   0, 1, 0)

        # Process the alpha signal
        current_alpha_signal = self.signal_handler.get_signal()
        if not self.initialized:
            self.previous_alpha_signal = current_alpha_signal
            self.initialized = True
        alpha_signal = py5.lerp(self.previous_alpha_signal, current_alpha_signal, self.smooth_factor)
        self.previous_alpha_signal = alpha_signal

        # Define generation ranges based on alpha signal
        generation_ranges = {
            1: (8.0, 9.0),
            2: (9.0, 10.0),
            3: (10.0, 11.0),
            4: (11.0, 13.0)
        }

        # Determine the number of generations based on alpha signal
        new_generations = 1
        for gen, (min_val, max_val) in generation_ranges.items():
            if min_val <= alpha_signal <= max_val:
                new_generations = gen
                break

        # Update L-system string if generations changed
        if new_generations != self.previous_generations:
            self.previous_generations = new_generations

        # Get the precomputed L-system for the current generation
        lsystem_string, self.branch_lengths = self.precomputed_lsystems[new_generations]

        # Calculate rotation and growth factor
        rotation_angle = (time.time() - self.start_time) * self.rotation_speed
        current_min, current_max = generation_ranges[new_generations]
        growth_factor = py5.remap(alpha_signal, current_min, current_max, 0, 1)
        growth_factor = py5.constrain(growth_factor, 0, 1)

        # Set background and apply transformations
        paper_color = self.config['colors']['paperColor']
        py5.background(*paper_color)
        py5.translate(py5.width / 2, 0)
        py5.scale(1, -1, 1)
        #py5.rotate_x(rotation_angle * 0.01)
        py5.rotate_y(rotation_angle * 0.01)
        #py5.rotate_z(rotation_angle * 0.02)

        # Draw the L-system
        self.draw_lsystem(lsystem_string, growth_factor)

    def generate_lsystem(self, generations):
        """Generate the L-system string based on the current number of generations."""
        lsystem_string = "F"
        branch_lengths = []
        current_branch_length = self.distance_branches

        for _ in range(generations):
            next_sentence = ""
            for current in lsystem_string:
                found = False
                for rule in self.rules:
                    if current == rule["a"]:
                        found = True
                        next_sentence += rule["b"]
                        for char in rule["b"]:
                            if char == 'F':
                                branch_lengths.append(current_branch_length)
                        break
                if not found:
                    next_sentence += current
                    if current == "F":
                        branch_lengths.append(current_branch_length)

            lsystem_string = next_sentence
            current_branch_length *= 0.5

        print(lsystem_string)
        return lsystem_string, branch_lengths

    def draw_lsystem(self, instructions, growth_factor):
        """Draw the L-system based on the generated instructions and growth factor."""
        branch_index = 0
        current_branch_length = 0
        ink_color1 = self.config['colors']['inkColor1']
        py5.stroke(*ink_color1)

        for current in instructions:
            if current == 'F':
                #py5.stroke_weight(random.randint(2, 5))
                current_branch_length = self.branch_lengths[branch_index] * growth_factor
                #py5.line(0, 0, 0, 0, -current_branch_length, 0)
				#self.draw_curve_branch(current_branch_length)  
                self.draw_bezier_branch(current_branch_length)
                py5.translate(0, -current_branch_length, 0)
                branch_index += 1
            elif current == 'X':
                fa = Fall(0, 0)
                self.falls.append(fa)  # Append the fall to the list
                fa.fashow(ink_color1)
                # for fall in self.falls[:]:  # Copy to avoid modifying list while iterating
                #     fall.faupdate()  # Update fall position and alpha
                #     fall.fashow(ink_color1)    # Show the fall effect
                #     # Remove fall object if it's finished (i.e., alpha < 0)
                #     if fall.finished():
                #         self.falls.remove(fall)
            elif current == 'f':
                py5.translate(0, -current_branch_length, 0)
            elif current == '+':
                py5.rotate_z(py5.radians(self.angle))
            elif current == '-':
                py5.rotate_z(py5.radians(-self.angle))
            elif current == '^':
                py5.rotate_x(py5.radians(self.angle))
            elif current == '&':
                py5.rotate_x(py5.radians(-self.angle))
            elif current == '\\':
                py5.rotate_y(py5.radians(self.angle))
            elif current == '/':
                py5.rotate_y(py5.radians(-self.angle))
            elif current == '[':
                py5.push_matrix()
            elif current == ']':
                py5.pop_matrix()

    def draw_curve_branch(self, current_branch_length):
        """Draws a curved branch using curveVertex with a given branch length."""
        py5.begin_shape()

        # Generate random values for curvature and length
        rand_curvature = random.randint(2, 5)
        rand_branch_length = random.uniform(0, current_branch_length)

        # Define the control points for the curve
        control_points = [
            (0, 0, 0),  # Initial point, repeated to ensure smooth start
            (0, 0, 0),
            (rand_curvature, -rand_branch_length / 2, rand_curvature),
            (rand_curvature, -rand_branch_length, rand_curvature),
            (0, -current_branch_length, 0),  # End point, repeated to ensure smooth end
            (0, -current_branch_length, 0)
        ]

        # Draw the curve using curveVertex
        for point in control_points:
            py5.curve_vertex(*point)

        py5.end_shape()

    def draw_bezier_branch(self, current_branch_length):
        """Draws a curved branch using a Bezier curve with a given branch length."""

        # Start drawing the shape
        py5.begin_shape()

        # Random values for curvature and length
        rand_curvature = random.randint(2, 5)
        rand_branch_length = random.uniform(0, current_branch_length)

        # Define the starting point of the Bezier curve
        start_point = (0, 0, 0)
        py5.vertex(*start_point)

        # Define control points for the Bezier curve
        control_point_1 = (rand_curvature, -rand_branch_length / 2, rand_curvature)
        control_point_2 = (rand_curvature, -rand_branch_length, rand_curvature)

        # Define the end point of the Bezier curve
        end_point = (0, -current_branch_length, 0)

        # Draw the Bezier curve
        py5.bezier(*start_point, *control_point_1, *control_point_2, *end_point)

        # End drawing the shape
        py5.end_shape()