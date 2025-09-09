# src/signal_processing/signal_manager.py
# pylint: disable=no-member

import threading
from .signal_handler import SignalHandler
from .alpha_waves import generate_live_alpha_signal, receive_data_from_LSL, read_signal_from_file

class SignalManager:
    """
    Manages signal processing, including receiving signals from various sources
    (generated, OSC, LSL, or file) and updating the signal handler accordingly.
    """
    def __init__(self, config):
        """
        Initializes the SignalManager with a given configuration.

        Args:
            config (dict): Configuration dictionary containing signal processing settings.
        """
        self.signal_handler = SignalHandler(
            config.get("min_power", 0.0),
            config.get("max_power", 50.0),
            8, 13
        )
        self.config = config

    def start_signal_processing(self):
        """
        Starts signal processing based on the specified source in the configuration.
        Launches threads to handle the signal input.
        """
        signal_source = self.config.get("signal_source", {})
        common_params = {
            "min_power": self.config.get("min_power", 0.0),
            "max_power": self.config.get("max_power", 50.0),
            "sample_rate": self.config.get("sample_rate", 0.1)
        }

        if "generated" in signal_source:
            print("Using generated signal")
            generated_config = signal_source["generated"]
            duration = generated_config.get("duration", 120)
            osc_ip = generated_config.get("osc_ip", "127.0.0.1")
            osc_port = generated_config.get("osc_port", 12346)
            osc_address = generated_config.get("osc_address", "/openbci/band-power/2")
            power_range = (common_params["min_power"], common_params["max_power"])
            sample_rate = common_params["sample_rate"]
            # Start OSC signal reception and generation threads
            threading.Thread(target=self.signal_handler.start_receiving,
                args=(osc_ip, osc_port, osc_address), 
                daemon=True
            ).start()
            threading.Thread(target=generate_live_alpha_signal,
                args=(duration, osc_ip, osc_port, osc_address, power_range, 1.0, sample_rate),
                daemon=True
            ).start()


        elif "osc" in signal_source:
            print("Using signal from OSC")
            osc_config = signal_source["osc"]
            threading.Thread(target=self.signal_handler.start_receiving, args=(
                osc_config.get("osc_ip", "127.0.0.1"),
                osc_config.get("osc_port", 12346),
                osc_config.get("osc_address", "/openbci/band-power/2")
            ), daemon=True).start()

        elif "lsl" in signal_source:
            print("Using LSL signal")
            threading.Thread(target=self.process_lsl_data, daemon=True).start()

        elif "file" in signal_source:
            print("Using pre-recorded signal from file")
            threading.Thread(target=self.process_file_data, daemon=True).start()

    def process_lsl_data(self):
        """
        Continuously receives alpha power data from LSL (Lab Streaming Layer) and 
        updates the signal handler.
        """
        for alpha_power in receive_data_from_LSL(self.config.get("sample_rate", 0.1)):
            self.signal_handler.update_signal(alpha_power)

    def process_file_data(self):
        """
        Reads pre-recorded signal data from a file and updates the signal handler accordingly.
        """
        file_path = self.config["signal_source"]["file"].get("file_signal_path", "Signal.txt")
        duration = self.config["signal_source"]["file"].get("duration", 120)
        for alpha_power in read_signal_from_file(file_path, duration, self.config.get("sample_rate", 0.1), lambda x: self.remap(x, -2, 2, self.config.get("min_power", 0.0), self.config.get("max_power", 50.0))):
            self.signal_handler.update_signal(alpha_power)
    
    def remap(self, value, in_min, in_max, out_min, out_max):
        """
        Remaps a value from one range to another.

        Args:
            value (float): The input value to remap.
            in_min (float): Minimum value of the input range.
            in_max (float): Maximum value of the input range.
            out_min (float): Minimum value of the output range.
            out_max (float): Maximum value of the output range.
        
        Returns:
            float: The remapped value in the output range.
        """
        return (value - in_min) / (in_max - in_min) * (out_max - out_min) + out_min
