# src/visuals/signal_visualization/draw_signal_manager.py
# pylint: disable=no-member

from visuals.signal_visualization.test_alpha_curve import AlphaWavePlotter

def start_draw_signal(signal_handler, draw_signal_cfg):
    """
    Reads configuration and launches the AlphaWavePlotter sketch.

    Args:
        signal_handler: The handler providing real-time signal data.
        draw_signal_cfg: A dict with keys:
            - enabled (bool): whether to launch the sketch
            - min_range_viz (float): min signal value for mapping
            - max_range_viz (float): max signal value for mapping
    """
    # Extract values with defaults
    enabled = draw_signal_cfg.get("enabled", True)
    if not enabled:
        return

    min_range = draw_signal_cfg.get("min_range_viz", 8)
    max_range = draw_signal_cfg.get("max_range_viz", 13)

    window_x  = draw_signal_cfg.get("window_x", 0)
    window_y  = draw_signal_cfg.get("window_y", 0)

    print(f"draw signal (min: {min_range}, max: {max_range})")

    # Instantiate and run the sketch
    plotter = AlphaWavePlotter(signal_handler, min_range=min_range, max_range=max_range, window_x=window_x, window_y=window_y)
    # Non-blocking run (requires recent py5)
    plotter.run_sketch(block=False)