# src/signal_processing/signal_handler.py
# pylint: disable=no-member
from pythonosc import dispatcher, osc_server
import queue
import threading

class SignalHandler:
    """
    A class to handle receiving and processing alpha wave signals via the OSC (Open Sound Control) protocol.

    Key Features:
    - Listens for OSC messages on a specified IP address, port, and address pattern.
    - Processes the alpha wave power to produce a frequency value within a target range.
    - Provides real-time updates of the latest signal value received.
    - Includes a method to scale signals independently for additional processing.

    Usage:
    1. Create an instance of SignalHandler.
    2. Call `start_receiving()` to begin listening for incoming OSC messages.
    3. Use `get_signal()` to access the latest processed signal value.
    """
    
    def __init__(self, min_power, max_power, min_range_viz, max_range_viz):
        """
        Initializes the SignalHandler with default values for signal handling and scaling.
        """
        self.current_signal = 0.0  # Store the most recent scaled signal value
        self.min_power = min_power
        self.max_power = max_power
        self.min_range_viz = min_range_viz
        self.max_range_viz = max_range_viz
        self.print_signal = True
        self.visual_trigger = 0
        self.trigger_listeners = []

        # Queue to store incoming signals
        self.signal_queue = queue.Queue(maxsize=100)
        self.running = True
        
        # Start a background thread to process signals
        self.processing_thread = threading.Thread(target=self.process_signals, daemon=True)
        self.processing_thread.start()

    def add_visual_trigger_listener(self, callback):
        """Registers a callback function to be called when visual trigger changes."""
        self.trigger_listeners.append(callback)

    def osc_handler(self, addr, *args):
        """
        Handles incoming OSC messages and processes the alpha wave signal.

        Args:
            addr (str): The OSC address of the incoming message.
            *args: Additional arguments, where the third argument (args[2]) is treated as the alpha wave power.
        """
        try:
            # Extract the alpha wave power from args[2]
            alpha_power = float(args[2])

            # Extract min and max
            self.min_power = float(args[0]) if args[0] is not None else self.min_power
            self.max_power = float(args[1]) if args[1] is not None else self.max_power

            # Scale the alpha power using the scaling function
            scaled_power = self.scale_signal(alpha_power)

            # Optionally log the processed values for debugging
            if self.print_signal:
                print(
                    f"Received Alpha Wave Power: {alpha_power:.2f}, "
                    f"Scaled Power: {scaled_power:.2f}"
                    )
            # Update the current signal with the scaled value
            self.signal_queue.put(scaled_power)

            if len(args) > 3:
                new_trigger = int(args[3])
                if new_trigger != self.visual_trigger:  # Only notify if state changes
                    self.visual_trigger = new_trigger
                    for callback in self.trigger_listeners:
                        callback(self.visual_trigger)
        
        except (ValueError, TypeError) as e:
            print(f"Error processing OSC message: {e}")

    def scale_signal(self, alpha_power):
        """
        Scales the raw alpha wave signal to the desired frequency range.

        Args:
            alpha_power (float): The raw alpha wave power.
            min_power (float): Minimum expected raw power value.
            max_power (float): Maximum expected raw power value.

        Returns:
            float: The scaled alpha wave signal in the desired range.
        """
        # Normalize the power to [0, 1]
        normalized_power = 50
        if self.max_power - self.min_power != 0:
            normalized_power = (alpha_power - self.min_power) / (self.max_power - self.min_power)
        normalized_power = max(0.0, min(normalized_power, 1.0))  # Clamp to [0, 1]

        # Scale to the visualization range
        scaled_power = normalized_power * (self.max_range_viz - self.min_range_viz) + self.min_range_viz
        return scaled_power

    def start_receiving(self, ip="127.0.0.1", port=5005, address="/alpha", print_signal=False):
        """
        Starts the OSC server to receive alpha wave signals.

        Args:
            ip (str): The IP address to listen on.
            port (int): The port to listen on.
            address (str): The OSC address pattern to listen for.
            min_range_viz (float): Minimum range for scaling the visualization.
            max_range_viz (float): Maximum range for scaling the visualization.
        """
        self.print_signal = print_signal
        
        # Set up the OSC dispatcher and map the address to the handler function
        disp = dispatcher.Dispatcher()
        disp.map(address, self.osc_handler)

        # Create and start the OSC server
        server = osc_server.BlockingOSCUDPServer((ip, port), disp)
        print(f"Serving on {ip}:{port} at address '{address}'")
        try:
            server.serve_forever()  # Start the server loop
        except KeyboardInterrupt:
            print("\nServer stopped by user.")

    def get_signal(self):
        """Fetch the latest signal from the queue if available, otherwise return the last known value."""
        try:
            self.current_signal = self.signal_queue.get_nowait()
        except queue.Empty:
            pass  # Keep the last known value if no new signals are available
        return self.current_signal
    
    def get_visual_trigger(self):
        """Returns the current state of the visual trigger (0 or 1)."""
        return self.visual_trigger
        
    def update_signal(self, alpha_power, print_signal=False):
        """
        Update the current signal value with an external alpha power reading.

        Args:
            alpha_power (float): The alpha wave power to process and update.
        """
        scaled_power = self.scale_signal(alpha_power)
        self.print_signal = print_signal
        if self.print_signal:
            print(
                f"Alpha Wave Power: {alpha_power:.2f}, "
                f"Scaled Power: {scaled_power:.2f}"
            )
        self.signal_queue.put(scaled_power)
        self.current_signal = scaled_power


    def process_signals(self):
        """
        Continuously fetches signals from the queue and updates self.current_signal.
        """
        while self.running:
            try:
                self.current_signal = self.signal_queue.get(timeout=1)  # Fetch latest signal
            except queue.Empty:
                pass  # No new signals, continue loop

    def stop(self):
        """Stops the signal processing thread."""
        self.running = False
        self.processing_thread.join()

# Example usage
if __name__ == "__main__":
    # Create an instance of the SignalHandler
    signal_handler = SignalHandler(0.0, 50.0, 8, 13)
    # Start listening for OSC messages
    signal_handler.start_receiving()
