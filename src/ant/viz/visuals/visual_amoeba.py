# src/visuals/visual_amoeba.py
# pylint: disable=no-member
import py5
from visuals.amoeba.amoeba import Amoeba
from .visual_base import VisualBase

EXTERNAL_RADIUS = 220
NUM_AMOEBAS = 30
SPEED_AMOEBAS = 150

OK_ALPHA = 9.0
BAD_ALPHA = 13.0

theta = 0

class VisualAmoeba(VisualBase):
    def __init__(self, signal_handler, config):
        super().__init__(signal_handler, config)
        self.shader = None
        self.signal_handler = signal_handler  # Store the signal handler
        self.config = config
        self.amoebas: list[Amoeba] = []
        self.must_align = False
        self.initialized = False
        self.color = self.get_color("color2")

    def init_spaw_amoebas(self):
        X = int(EXTERNAL_RADIUS)
        step = 100
        idx = 0
        for x in range(-X, X+1, step):
            Y = int( (EXTERNAL_RADIUS*EXTERNAL_RADIUS - x*x)**0.5 ) # bound for y given x
            for y in range(-Y, Y+1, step):
                a = self.spawn_amoeba(idx, x, y, 69, 99)
                self.amoebas.append(a)
                idx += 1
        while len(self.amoebas) <= NUM_AMOEBAS:
            a = self.try_to_spawn_amoeba(idx, X, 39, 59)
            if a:
                self.amoebas.append(a)
                idx += 1

    def try_to_spawn_amoeba(self, idx, X, min_d, max_d, fade_in = False):
        x = py5.random(-X, X)
        Y = int( (EXTERNAL_RADIUS*EXTERNAL_RADIUS - x*x)**0.5 ) # bound for y given x
        y = py5.random(-Y, Y+1)
        a = self.spawn_amoeba(idx, x, y, min_d, max_d, fade_in)
        tries = 0
        while self.collides_with_any_other(a) and tries < 200:
            x = py5.random(-X, X)
            Y = int( (EXTERNAL_RADIUS*EXTERNAL_RADIUS - x*x)**0.5 ) # bound for y given x
            y = py5.random(-Y, Y+1)
            a.set_position(x, y)
            tries += 1
        if not self.collides_with_any_other(a):
            return a

    def spawn_amoeba(self, idx, x, y, min_d, max_d, fade_in = False):
        diameter = py5.random(min_d, max_d)
        speed = SPEED_AMOEBAS / (diameter * 80)
        return Amoeba(idx, x, y, diameter, speed, self.color, fade_in = fade_in)

    def collides_with_any_other(self, a: Amoeba):
        result = False
        for b in self.amoebas:
            if a is b:
                continue
            if a.collides_with(b):
                result = True
        return result

    def handle_good_signal(self):
        self.set_canvas_rotate()
        next_idx = len(self.amoebas) + 1
        if len(self.amoebas) < 50:
            a = self.try_to_spawn_amoeba(next_idx, EXTERNAL_RADIUS, 69, 99, True)
            if not a:
                a = self.try_to_spawn_amoeba(next_idx, EXTERNAL_RADIUS, 39, 59, True)
            if not a:
                a = self.try_to_spawn_amoeba(next_idx, EXTERNAL_RADIUS, 29, 39, True)
            if a:
                self.amoebas.append(a)
        for a in self.amoebas:
            a.set_synchronized_movement()

    def handle_bad_signal(self):
        self.set_canvas_stop_rotate()
        num_remaining = sum(1 for a in self.amoebas if not a.fade_out)
        if num_remaining > 20:
            delete_rand = int(py5.random(0, len(self.amoebas)))
            self.amoebas[delete_rand].fade_out = True
        for a in self.amoebas:
            a.set_free_movement()

    def handle_collision_with_others(self, a_idx, a: Amoeba):
        for b in self.amoebas[a_idx:]:
            if a is b:
                continue
            if a.collides_with(b):
                a.elastic_collision(b)

    def set_free_movement(self):
        self.set_canvas_stop_rotate()
        for a in self.amoebas:
            a.set_free_movement()

    def handle_collision_with_external_radius(self, a: Amoeba):
        center_vec = py5.Py5Vector2D(0, 0)
        pos_vec = center_vec - a.position
        if pos_vec._get_mag() > EXTERNAL_RADIUS and not a.has_bounced_external_radius:
            a.velocity = -a.velocity
            a.has_bounced_external_radius = True
        if pos_vec._get_mag() > EXTERNAL_RADIUS - a.radius:
            a.has_bounced_external_radius = False

    def set_canvas_rotate(self):
        self.must_align = True

    def set_canvas_stop_rotate(self):
        global theta
        self.must_align = False

    def draw(self):
        global theta
        py5.no_stroke()
        if not self.initialized:
            self.init_spaw_amoebas()
            self.initialized = True
        py5.push_matrix()
        py5.no_fill()
        X_CENTER = int(py5.width / float(2))
        Y_CENTER = int(py5.height / float(2))
        py5.translate(X_CENTER, Y_CENTER)

        if self.must_align:
            theta -= 0.0025
        py5.rotate(theta)
        background_color = self.get_backgroundcolor("backgroundColor2")
        py5.background(*background_color)
        
        frame_rate = py5.get_frame_rate()
        dt = frame_rate / 60
        alpha_signal = self.signal_handler.get_signal()
        
        if py5.random(1, 100) <= 5:
            if alpha_signal <= OK_ALPHA:
                self.handle_good_signal()
            elif alpha_signal >= BAD_ALPHA:
                self.handle_bad_signal()
            else:
                self.set_free_movement()

        for a_idx, a in enumerate(self.amoebas):
            if a.alpha <= 1:
                self.amoebas.remove(a)
                continue
            self.handle_collision_with_others(a_idx, a)
            self.handle_collision_with_external_radius(a)
                
        for a in self.amoebas:
            a.move(dt)
            a.prepare_display()

        for a in self.amoebas:
            a.display()
        py5.pop_matrix()  # Restore transformation matrix
        super().draw()
