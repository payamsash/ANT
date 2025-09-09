# src/visuals/tree/leaf.py
# pylint: disable=no-member
from random import randint
import py5

class Leaf:    
    def __init__(self, params):
        # Randomize the size of the leaf
        self.bloom_size = py5.random(params['bloomSizeAverage'] * 0.7, params['bloomSizeAverage'] * 1.3)
        self.bloom_ratio = params['bloomWidthRatio']

        # Randomize the leaf's rotation
        self.leaf_rot = py5.radians(py5.random(-180, 180))

        # Random delay before the leaf starts blooming
        self.leaf_delay = randint(50, 150)

        # Initial scale of the leaf (it starts at 0 and grows)
        self.leaf_scale = 0.0

        # Initial transparency (alpha) of the leaf
        self.alpha = py5.random(200, 255)
        
        # Randomly select a leaf color from the provided options in params
        color1, color2, color3 = params['colors']
        self.leaf_color = py5.color(*color1) if py5.random(1) < 0.33 else (
                          py5.color(*color2) if py5.random(1) < 0.66 else py5.color(*color3))

        # Set up initial falling properties
        self.falling = False


    def update_falling(self):
        """
        Updates the position, rotation, and transparency of the leaf as it falls. Simulates gravity, wind, and rotation during the fall.
        """
        if self.falling and self.alpha > 0:
            # Fade out alpha over time
            self.alpha = max(0, self.alpha - 5)

            # Stop falling if fully transparent
            if self.alpha < 1:
                self.falling = False


    def init_falling(self):
        """
        If the leaf isn't falling, it will start the falling process.
        """
        self.falling = True

    def draw(self):
        """
        Draws the leaf on the screen at its current position, applying the bloom size, rotation, and transparency.
        If the leaf is falling, it will be drawn at its falling position; otherwise, it will be drawn at the bloom position.
        """
        py5.fill(self.leaf_color, self.alpha)  # Set the fill color with transparency
        py5.no_stroke()  # Disable the stroke (outline) for the leaf
        py5.push_matrix()

        py5.translate(0, -self.bloom_size / 2)

        self.leaf_scale += (1.0 - self.leaf_scale) * 0.01
        py5.scale(self.leaf_scale)

        # Apply the leaf's rotation
        py5.rotate(self.leaf_rot)

        # Draw the leaf as an ellipse
        py5.ellipse(0, 0, self.bloom_size * self.bloom_ratio, self.bloom_size)

        py5.pop_matrix()