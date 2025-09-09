# src/visuals/signal_visualization/test_alpha_curve.py
# pylint: disable=no-member
import time
import py5

class AlphaWavePlotter(py5.Sketch):
    """
    A Py5 sketch that visualizes alpha brainwave signals over time.
    The visualization updates dynamically and smooths the signal to
    provide a clearer representation of the data.
    
    Attributes:
        signal_handler: An object that provides real-time alpha wave signal values.
        min_range (float): Minimum expected signal value for scaling.
        max_range (float): Maximum expected signal value for scaling.
        start_time (float): Time at which the sketch started running.
        data (list): A list of (time, signal_value) tuples for plotting.
        duration (int): The time window (in seconds) for displaying recent signals.
        window_x (int): X coordinate of the window on screen.
        window_y (int): Y coordinate of the window on screen.
    """
    def __init__(self, signal_handler, min_range, max_range, window_x=0, window_y=0):
        super().__init__()
        self.signal_handler = signal_handler
        self.min_range = min_range
        self.max_range = max_range
        self.start_time = time.time()
        self.data = []
        self.duration = 10  # Display only the last 10 seconds of data
        self.pos_window_x = window_x
        self.pos_window_y = window_y

    def settings(self):
        """Initializes the Py5 canvas size and sets an application icon."""
        self.size(500, 400, py5.P2D)
        try:
            PJOGL = py5.JClass("processing.opengl.PJOGL")
            PJOGL.setIcon("assets/images/epflecal_lab-logo.png")
        except Exception as e:
            print("Warning: could not set window icon:", e)

    def setup(self):
        """Configures initial sketch settings, including background color and frame rate."""
        self.background(255)
        self.frame_rate(60)
        self.window_move(self.pos_window_x, self.pos_window_y)
        self.window_title("Alpha Waves")

    def draw(self):
        """
        Main Py5 loop that updates the visualization by retrieving new signal values,
        smoothing the data, and rendering the graph.
        """
        self.background(255)
        current_time = time.time() - self.start_time
        x_start, x_end = (0, current_time) if current_time < self.duration else (current_time - self.duration, current_time)
        self.draw_axes(x_start, x_end)

        # Get the current signal value from the handler.
        signal_value = self.signal_handler.get_signal()

        # Store data (only keeping points within the last 'duration' seconds)
        self.data.append((current_time, signal_value))
        self.data = [(t, s) for t, s in self.data if t >= current_time - self.duration]
        smoothed_data = self.smooth_signal(self.data)

        # Define min/max range for mapping signal values to screen coordinates
        y_min, y_max = self.min_range, self.max_range

        # Draw the signal curve
        self.stroke(255, 0, 0)  # Red line for signal plot
        self.no_fill()
        self.begin_shape()
        for t, s in smoothed_data:
            x = py5.remap(t, x_start, x_end, 50, self.width - 20)
            y = py5.remap(s, y_min, y_max, self.height - 30, 20)
            self.vertex(x, y)
        self.end_shape()

    def smooth_signal(self, data, window_size=10):
        """
        Applies a simple moving average (SMA) filter to smooth signal fluctuations.
        
        Args:
            data (list of tuples): List of (time, signal_value) points.
            window_size (int): Number of points to average for smoothing.

        Returns:
            list of tuples: Smoothed signal data.
        """
        if len(data) < window_size:
            return data  # Not enough points to smooth
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size + 1)
            values = [s for _, s in data[start:i + 1]]
            smoothed.append((data[i][0], sum(values) / len(values)))
        return smoothed

    def draw_axes(self, x_start, x_end):
        """
        Draws the X and Y axes along with tick marks and dynamic labels.
        
        Args:
            x_start (float): The starting time of the X-axis.
            x_end (float): The ending time of the X-axis.
        """
        self.stroke(0)
        self.stroke_weight(2)
        
        # Draw X-axis
        self.line(50, self.height - 30, self.width - 20, self.height - 30)
        tick_interval = 1
        first_tick = x_start - (x_start % tick_interval)
        if first_tick < x_start:
            first_tick += tick_interval

        t = first_tick
        while t <= x_end:
            x = py5.remap(t, x_start, x_end, 50, self.width - 20)
            self.line(x, self.height - 35, x, self.height - 25)  # Draw tick mark.
            self.fill(0)
            self.text(f"{t:.1f}s", x - 10, self.height - 10)
            t += tick_interval

        # Draw Y-axis
        self.line(50, 20, 50, self.height - 30)
        
        # Draw Y-axis ticks between min_range and max_range.
        for i in range(self.min_range, self.max_range + 1):
            y = py5.remap(i, self.min_range, self.max_range, self.height - 30, 20)
            self.line(48, y, 52, y)  # Tick mark.
            self.fill(0)
            self.text(f"{i}", 30, y + 5)