# src/visual_manager.py
# pylint: disable=no-member

class VisualManager:
    """
    Manages the selection and switching of visual presets.
    
    This class allows loading a single or multiple visual presets, cycling through them,
    and retrieving the currently active preset. It also supports an optional event callback
    that triggers when the preset changes.
    """
    def __init__(self, visuals_preset, all_visuals, event_callback=None):
        self.current_preset = None
        self.available_presets = all_visuals
        self.preset_queue = []
        self.load_preset(visuals_preset)
        self.event_callback = event_callback  # Event callback for preset changes

    def load_preset(self, visuals_preset):
        """Load a preset or multiple presets if they exist in all_visuals."""
        valid_presets = []

        if isinstance(visuals_preset, str):
            if visuals_preset in self.available_presets:
                valid_presets.append(visuals_preset)
            else:
                print(f"Warning: Preset '{visuals_preset}' not found. Using default preset.")
                valid_presets.append("VisualRorschach")

        elif isinstance(visuals_preset, list):
            valid_presets = [p for p in visuals_preset if p in self.available_presets]

            if not valid_presets:  # If no valid presets were found
                print("Warning: No valid presets found in the list. Using default preset.")
                valid_presets.append("VisualRorschach")

        self.preset_queue = valid_presets
        self.current_preset = self.preset_queue[0]

    def switch_preset(self):
        """Cycle through presets if multiple presets are given."""
        if len(self.preset_queue) > 1:
            self.preset_queue.append(self.preset_queue.pop(0))  # Rotate presets
            self.current_preset = self.preset_queue[0]
            if self.event_callback:
                self.event_callback(self.current_preset)

    def get_current_preset(self):
        """Return the active preset."""
        return self.current_preset