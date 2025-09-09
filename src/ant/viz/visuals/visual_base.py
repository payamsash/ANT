# src/visuals/visual_base.py
# pylint: disable=no-member

from abc import ABC, abstractmethod
from typing import Any, Dict
import re
import py5
import psutil
import gc
import time
from version import __version__

show_version = False
show_fps = False
show_memory_usage = False
show_signal = False
memory_info = None

class VisualBase(ABC):
    """
    Abstract base class for creating visual presets.

    This class serves as a blueprint for all visual presets in the project.
    It defines a standard structure that all derived visual classes must follow.

    Attributes:
        signal_handler (Any): The object responsible for handling incoming signals.
        config (Dict[str, Any]): A dictionary containing configuration parameters.
    """
    def __init__(self, signal_handler: Any, config: Dict[str, Any]) -> None:
        """
        Initializes the VisualBase class with a signal handler and configuration.

        Args:
            signal_handler (Any): An instance responsible for managing signal inputs.
            config (Dict[str, Any]): A dictionary of configuration parameters for the visual.
        """
        self.signal_handler = signal_handler
        self.config = config
        self.last_memory_check = time.time()

    def settings(self) -> None:
        """
        Configures the rendering settings for the visual.
        This method is final and cannot be overridden by subclasses.
        """
        self.renderer = py5.P3D
        is_full_screen = self.config.get("is_full_screen", True)
        if is_full_screen:
            py5.full_screen(self.renderer)
        else:
            py5.size(800, 800, self.renderer)
            PJOGL = py5.JClass("processing.opengl.PJOGL")
            PJOGL.setIcon("assets/images/epflecal_lab-logo.png")
        py5.smooth(4)

    def setup(self) -> None:
        """
        Sets up the initial state for the visual.
        This method is final and cannot be overridden by subclasses.
        """
        surface = py5.get_surface()
        surface.set_always_on_top(True)
        surface.set_location(0, 0)
        show_cursor = self.config.get("show_cursor", True)
        if show_cursor:
            py5.cursor()
        else:
            py5.no_cursor()
        class_name = self.__class__.__name__
        formatted_name = re.sub(r'([A-Z])', r' \1', class_name).strip()
        surface.set_title(f"Advancing Neurofeedback in Tinnitus - {formatted_name}")

    def draw(self) -> None:
        """
        Renders the visual frame. This method could be overridden by subclasses.
        """
        global show_version, show_fps, show_memory_usage, memory_info, show_signal
        py5.push_matrix()
        if show_version:
            py5.stroke(0)
            py5.text_size(16)
            py5.fill(0)
            py5.text(f"Version: {__version__}", 10, py5.height - 10)

        if show_fps:
            fps = py5.get_frame_rate()
            py5.stroke(0)
            py5.text_size(16)
            py5.fill(0)
            py5.text(f"FPS: {fps:.2f}", py5.width - 100, py5.height - 10)

        if show_memory_usage:
            if time.time() - self.last_memory_check > 1:
                memory_info = psutil.Process().memory_info().rss / (1024 * 1024)
                self.last_memory_check = time.time()
            py5.stroke(0)
            py5.text_size(16)
            py5.fill(0)
            py5.text(f"Memory: {memory_info:.2f} MB", 10, py5.height - 30)
        if show_signal:
            py5.stroke(0)
            py5.text_size(16)
            py5.fill(0)
            py5.text(f"Signal: {self.signal_handler.get_signal():.2f} ", py5.width - 100, py5.height - 30)
        py5.pop_matrix()
         
    def key_pressed(self, e) -> None:
        """Handles key press events for toggling version and FPS display."""
        global show_version, show_fps, show_memory_usage, show_signal
        if e.is_control_down():
            if py5.key_code == ord('V'):
                show_version = not show_version
            elif py5.key_code == ord('F'):
                show_fps = not show_fps
            elif py5.key_code == ord('M'):
                show_memory_usage = not show_memory_usage
            elif py5.key_code == ord('S'):
                show_signal = not show_signal

    def get_color(self, *keys):
        """
        Retrieves a color from the configuration, prioritizing `universalColor`.
        """
        colors = self.config.get("colors", {})
        universal_color = colors.get("universalColor", [])
        if universal_color:
            return universal_color
        return next((colors.get(k) for k in keys if k in colors), None)
    
    def get_backgroundcolor(self, *keys):
        """
        Retrieves a background color from the configuration, prioritizing `universalBackgroundColor`.
        """
        colors = self.config.get("colors", {})
        universal_backgroundcolor = colors.get("universalBackgroundColor", [])
        if universal_backgroundcolor:
            return universal_backgroundcolor
        return next((colors.get(k) for k in keys if k in colors), None)

    def cleanup(self):
        """Frees resources and prepares for visual transition."""
        if hasattr(self, "signal_handler"):
            self.signal_handler = None  # Drop reference to avoid memory leaks
        if hasattr(self, "config"):
            self.config.clear()  # Clear the dictionary (optional)
        gc.collect()