import numpy as np
from abc import ABC, abstractmethod
from constants import CAR_LENGTH, CAR_WIDTH, INITIAL_WALL_OFFSET, WALL_VELOCITY, WALL_COMPARE

START_VELOCITY = 0
MAX_VELOCITY = 20


class Car(ABC):
    def __init__(self, x, y, v_x=START_VELOCITY, v_y=START_VELOCITY):
        # Setting Initial Values
        self.fin = False
        self.x = x
        self.y = y
        self.v_x = v_x
        self.v_y = v_y
        self.rotation = 0
        self.wall_position = 0

    def hit_wall(self):
        return self.wall_distance <= -WALL_COMPARE

    @abstractmethod
    def finished(self):
        pass

    @property
    def wall_distance(self):
        return 0

    def update_position(self, accel, rotation, time_step):
        if self.fin:
            return
        self.wall_position += WALL_VELOCITY * time_step
        self.rotation = (self.rotation + rotation) % (2*np.pi)
        #print("Car accel", accel, "and rotation", self.rotation)
        # print(self.rotation)
        # print(np.cos(self.rotation))
        # print((np.sin(self.rotation) * (-1)))
        # print(np.cos(self.rotation))
        a_x = accel * np.sin(self.rotation) * (-1)
        a_y = accel * np.cos(self.rotation)

        if self.v_x < MAX_VELOCITY:
            self.v_x += a_x * time_step

        if self.v_y < MAX_VELOCITY:
            self.v_y += a_y * time_step

        #print("Velocty X", self.v_x)
        #print("Velocty Y", self.v_y)

        self.x += self.v_x * time_step
        self.y += self.v_y * time_step

    def points(self):
        '''for rendering in run_sim'''
        return [
            (
                np.cos(self.rotation) * (CAR_WIDTH / 2) - np.sin(self.rotation) * (CAR_LENGTH / 2) + self.x,
                np.sin(self.rotation) * (CAR_WIDTH / 2) + np.cos(self.rotation) * (CAR_LENGTH / 2) + self.y,
            ),
            (
                np.cos(self.rotation) * (CAR_WIDTH / 2) - np.sin(self.rotation) * (-CAR_LENGTH / 2) + self.x,
                np.sin(self.rotation) * (CAR_WIDTH / 2) + np.cos(self.rotation) * (-CAR_LENGTH / 2) + self.y,
            ),
            (
                np.cos(self.rotation) * (-CAR_WIDTH / 2) - np.sin(self.rotation) * (-CAR_LENGTH / 2) + self.x,
                np.sin(self.rotation) * (-CAR_WIDTH / 2) + np.cos(self.rotation) * (-CAR_LENGTH / 2) + self.y,
            ),
            (
                np.cos(self.rotation) * (-CAR_WIDTH / 2) - np.sin(self.rotation) * (CAR_LENGTH / 2) + self.x,
                np.sin(self.rotation) * (-CAR_WIDTH / 2) + np.cos(self.rotation) * (CAR_LENGTH / 2) + self.y,
            ),
        ]

class VerticalCar(Car):
    '''car moving vertically (in y direction)'''
    def __init__(self, x, y):
        super().__init__(x, y, v_y=5)
        self.wall_position = y - INITIAL_WALL_OFFSET

    @property
    def wall_distance(self):
        return self.y - self.wall_position

    def finished(self):
        return self.fin or self.y >= 40


class HorizontalCar(Car):
    '''car moving horizontally (in x direction)'''
    def __init__(self, x, y):
        super().__init__(x, y, v_x=5)
        self.rotation = 3 * np.pi / 2
        self.wall_position = x - INITIAL_WALL_OFFSET

    @property
    def wall_distance(self):
        return self.x - self.wall_position

    def finished(self):
        return self.fin or self.x >= 40

