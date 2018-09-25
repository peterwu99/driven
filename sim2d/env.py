import numpy as np
from constants import ENVS, TIME_STEP, CAR_RADIUS, ENVS
from car import VerticalCar, HorizontalCar
from random import randint


class Environment(object):
    def __init__(self):
        # Absolute Length of Road
        self.grid = ENVS["FOUR_WAY_1"]
        self.cars = []

        # List of Cars
        self.cars.append(VerticalCar(2, 5))
        self.cars.append(HorizontalCar(5, 20))
        self.cars.append(VerticalCar(20, 13))
        self.cars.append(HorizontalCar(13, 20))

    def update(self, accel_list, time_step=TIME_STEP):
        """
        Args:
            accel_list: list containing the acceleration to apply to the
                respective car in self.cars
            time_step: number of seconds occurring during this update
        """
        # Apply to Each Car
        for index in range(len(self.cars)):
            self.cars[index].update_position(accel_list[index][0], accel_list[index][1], time_step)

        # print("Car 1 x", self.cars[0].x, "y", self.cars[0].y)
        # print("Car 2 x", self.cars[1].x, "y", self.cars[1].y)

        if all(car.fin for car in self.cars):
            return 100 # todo need to scale based on number of cars

        # Resolving Cars that Finished
        # currently displaces finished cars an arbitrary far distance away
        for car in self.cars:
            if car.fin:
                car.x = 400
                car.y = 400
            elif car.finished():
                car.fin = True
                car.x = 400
                car.y = 400
                return sum(car.fin for car in self.cars)

        # Check Boundary Error
        for car in self.cars:
            round_x = int(car.x)
            round_y = int(car.y)
            # print("round x", round_x, "and round y", round_y)
            if round_x < 0 or round_y < 0:
                return -10
            if not car.fin and ENVS["FOUR_WAY_1"][round_x][round_y] == 0:
                return -1

        # Check Wall Crash
        for car in self.cars:
            if car.hit_wall():
                return -5

        # Check Car Crash
        for car in self.cars:
            for other_car in self.cars:
                if car == other_car:
                    continue

                if np.sqrt((other_car.x - car.x)**2 + (other_car.y - car.y)**2) < (2 * CAR_RADIUS):
                    return -1 # 0 # -1
        
        wall_dist_sum = np.sum([c.wall_distance for c in self.cars if not c.fin])
        # print(wall_dist_sum) # generally < 30, 
        wall_factor = 1# 3*np.tanh(wall_dist_sum)

        return int(np.sum([int(c.fin) for c in self.cars])*wall_factor) # + 0.1
