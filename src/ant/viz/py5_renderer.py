# src/rendering/visual_base.py
# pylint: disable=no-member
import py5
import psutil
import gc
import cProfile

check_memory = False

def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)  # Convert to MB

def check_garbage():
    gc.collect()
    print(f"Unreachable objects: {len(gc.garbage)}")

def profile_drawing(visual):
    profiler = cProfile.Profile()
    profiler.enable()
    visual.draw()   
    profiler.disable()
    profiler.print_stats(sort="cumulative")

def render_visual(visual, restart):
    if check_memory:
        print(f"Memory before loading visual: {get_memory_usage():.2f} MB")
    def settings():
        """Py5 settings function."""
        if hasattr(visual, 'settings') and callable(visual.settings):
            try:
                visual.settings()
            except Exception as e:
                print(f"Error in visual.settings(): {e}")

    def setup():
        """Py5 setup function."""
        if hasattr(visual, 'setup') and callable(visual.setup):
            try:
                visual.setup()
            except Exception as e:
                print(f"Error in visual.setup(): {e}")

    def draw():
        """Py5 draw function."""
        py5.background(255)  # Clear the frame with a white background
        if hasattr(visual, 'draw') and callable(visual.draw):
            try:
                #if not check_memory:
                    visual.draw()
                #else:
                    #profile_drawing(visual)  # Profile drawing performance
            except Exception as e:
                print(f"Error in visual.draw(): {e}")

    def key_pressed(e):
        """Handles key press events."""
        if hasattr(visual, 'key_pressed') and callable(visual.key_pressed):
            try:
                visual.key_pressed(e)
            except Exception as ex:
                print(f"Error in visual.key_pressed(): {ex}")

    # Assign the py5 lifecycle functions
    if restart:
        py5.hot_reload_draw(visual.draw)
    else:
        py5.settings = lambda: settings()
        py5.setup = lambda: setup()
        py5.draw = lambda: draw()
        py5.key_pressed = lambda e: key_pressed(e)
    try:
        if not restart:
            py5.run_sketch()
            
    except Exception as e:
        print(f"Error running py5 sketch: {e}")

    if check_memory:
        print(f"Memory after loading visual: {get_memory_usage():.2f} MB")
        check_garbage()