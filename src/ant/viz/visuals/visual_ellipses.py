# src/visuals/visual_ellipses.py
# pylint: disable=no-member
import py5
from .visual_base import VisualBase

theta = 0
OK_ALPHA = 9.0
MAX_ALPHA = 13.0

class VisualEllipses(VisualBase):
    def __init__(self, signal_handler, config):
        super().__init__(signal_handler, config)
        self.shader = None
        self.signal_handler = signal_handler  # Store the signal handler
        self.config = config
        self.last_signal = -1
        self.must_animate_tilt = False
        self.current_tilt = 0
        self.color = self.get_color("color2")
        self.tilt_to = 0
        self.tilt_amount = 0
        self.last_tilt = 0
        self.initialized = False

    def draw_ellipse(self, x, y, width, height, theta, tilt):
        py5.push_matrix()
        py5.translate(x, y)
        py5.rotate(theta + tilt)
        py5.ellipse(0, 0, width, height)
        py5.pop_matrix()

    def draw(self):
        if not self.initialized:  # Initialize smoothing on first frame
            py5.no_fill()
            py5.stroke_weight(0.8)
            py5.noise_seed(1020)
            py5.noise_detail(4, 0.5)
            self.initialized = True
        py5.stroke(*self.color)
        frame_rate = py5.get_frame_rate()
        dt = frame_rate / 60.0
        alpha_signal = self.signal_handler.get_signal()

        tilt_goal = py5.remap(alpha_signal, OK_ALPHA, MAX_ALPHA, 0, 3)
        diff_tilt = tilt_goal - self.current_tilt

        if tilt_goal <= 0.5 and diff_tilt > 0:
            if self.current_tilt <= 0.00001:
                self.current_tilt = 0
            else:
                self.current_tilt *= 1.01

        elif tilt_goal <= 0.5 and diff_tilt < 0:
            if self.current_tilt <= 0.00001:
                self.current_tilt = 0
            else:
                self.current_tilt /= 1.01
                
        else:
            self.current_tilt += diff_tilt * dt / 100

        global theta
        theta -= 0.0025

        width = 360
        height = 120
        separator = 10
        separator_divider = 2.0
        x = 10
        y = 0

        background_color = self.get_backgroundcolor('backgroundColor2')
        py5.background(*background_color)
        py5.push_matrix()
        py5.no_fill()
        py5.translate(py5.width//2, py5.height//2)
        py5.rotate(theta)

        for dist in range(1, 101):
            x += separator

            tilt = py5.noise(x, y, py5.frame_count / 100.0) * \
                py5.remap(dist, 0, 100, 0.33, 1) * \
                self.current_tilt

            if dist in [7, 13, 17, 25, 36, 49, 69, 71, 76, 80, 84, 86, 87, 90, 91, 96]:
                tilt /= 2.0

            self.draw_ellipse(x, y, width, height, theta, tilt)
            self.draw_ellipse(-x, -y, width, height, theta, tilt)

            separator_divider /= 1.8
            separator = max(separator/max(separator_divider, 1.1), 1.5)
        py5.pop_matrix()
        super().draw()

