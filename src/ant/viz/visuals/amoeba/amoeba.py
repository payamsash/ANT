import py5
import math

cs = py5.get_current_sketch()

# Source: https://py5coding.org/tutorials/intro_to_py5_and_python_17_object_oriented_programming.html
class Amoeba():
    def __init__(self, id, x, y, diameter, speed, color, fade_in = False):
        self.id = id
        self.position = py5.Py5Vector(x, y)
        self.speed = speed
        self.velocity = self.create_random_velocity(speed)
        self.radius = diameter / 2.0
        self.mass = diameter * math.pi
        self.must_align = False
        self.must_choose_random_velocity = False
        self.fade_out = False
        self.has_bounced_external_radius = False
        self.color = color
        self.fade_in = fade_in
        self.alpha = 5 if fade_in else 255 # must be > 1, otherwise it will be automatically deleted

    def create_random_velocity(self, speed):
        range_x = py5.random(2, 5)
        range_y = py5.random(2, 5)
        xdirection = -range_x if py5.random(1) < 0.5 else range_x
        ydirection = -range_y if py5.random(1) < 0.5 else range_y
        return py5.Py5Vector(speed * xdirection, speed * ydirection)

    def set_position(self, x, y):
        self.position = py5.Py5Vector(x, y)

    def set_synchronized_movement(self):
        self.must_align = True

    def set_free_movement(self):
        if self.must_align:
            self.must_choose_random_velocity = True
        self.must_align = False

    def circle_point(self, t, r):
        x = py5.cos(t) * r
        y = py5.sin(t) * r
        return [x, y]

    def move(self, dt):
        if self.must_align:
            return
        elif self.must_choose_random_velocity:
            self.velocity = self.create_random_velocity(self.speed)
            self.must_choose_random_velocity = False
        self.position += self.velocity * dt

    def collides_with(self, other):
        dist = math.sqrt(
            math.pow(self.position.x - other.position.x, 2) + math.pow(self.position.y - other.position.y, 2))
        minDistance = self.radius + other.radius
        return dist <= minDistance

    # Source: https://en.wikipedia.org/w/index.php?title=Elastic_collision#Two-dimensional_collision_with_two_moving_objects
    def elastic_collision(self, other):
        m1 = self.mass
        m2 = other.mass
        v1 = self.velocity
        v2 = other.velocity
        x1 = self.position
        x2 = other.position
        self.velocity = \
            v1 - (2 * m2 / (m1 + m2)) * \
                 (py5.Py5Vector.dot( (v1 - v2), (x1 - x2) ) / (x1 - x2)._get_mag_sq()) * \
                 (x1 - x2)
        other.velocity = \
            v2 - (2 * m1 / (m1 + m2)) * \
                 (py5.Py5Vector.dot( (v2 - v1), (x2 - x1)) / (x2 - x1)._get_mag_sq()) * \
                 (x2 - x1)

    def diff_angle_to(self, to: py5.Py5Vector):
        return self.velocity.dot(to)

    def prepare_display(self):
        if self.fade_in and self.alpha < 255:
            self.alpha += 1
        if self.fade_out and self.alpha > 0:
            self.alpha -= 1


    def display(self):
        py5.fill(*self.color, self.alpha)

        self_x = self.position.x
        self_y = self.position.y
        
        r = self.radius
        cpl = r * 0.55 
        cpx, cpy = self.circle_point(cs.frame_count/(r/2), r/8) 
        xp, xm = self_x+cpx, self_x-cpx
        yp, ym = self_y+cpy, self_y-cpy

        py5.begin_shape()
        py5.vertex(
            self_x, self_y-r # top vertex
        )
        py5.bezier_vertex(
            xp+cpl, yp-r, xm+r, ym-cpl,
            self_x+r, self_y # right vertex
        )
        py5.bezier_vertex(
            xp+r, yp+cpl, xm+cpl, ym+r,
            self_x, self_y+r # bottom vertex
        )
        py5.bezier_vertex(
            xp-cpl, yp+r, xm-r, ym+cpl,
            self_x-r, self_y # left vertex
        )
        py5.bezier_vertex(
            xp-r, yp-cpl, xm-cpl, ym-r,
            self_x, self_y-r # (back to) top vertex
        )
        py5.end_shape()