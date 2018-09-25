from constants import CAR_LENGTH, NUM_CARS, ROAD_LENGTH, TIME_STEP

START_VELOCITY = 0
INITIAL_WALL_OFFSET = 1
WALL_COMPARE = 10
WALL_VELOCITY = 5
MAX_VELOCITY = 20


class Car(object):
    def __init__(self, position, velocity=START_VELOCITY):
        # Setting Initial Values
        self.position = position
        self.wall_position = position - INITIAL_WALL_OFFSET
        self.velocity = velocity
        
    def hit_wall(self):
        return (self.position - self.wall_position) <= -WALL_COMPARE

    def update_position(self, accel, time_step):
        if self.velocity < MAX_VELOCITY:
            self.velocity += accel * time_step
        
        self.position += self.velocity * time_step
        self.wall_position += WALL_VELOCITY * time_step


class Environment(object):
    def __init__(self, num_of_cars=NUM_CARS):
        # Absolute Length of Road
        self.length = ROAD_LENGTH
        self.cars = []

        # List of Cars
        for car in range(num_of_cars):
            self.cars.append(Car(INITIAL_WALL_OFFSET + (CAR_LENGTH + INITIAL_WALL_OFFSET)*car))

        # For Checking Cars Passing Finish Line
        self.finished_cars = [False for _ in self.cars]

    def update(self, accel_list, time_step=TIME_STEP, cycle=False):
        '''
        Args:
            accel_list: list containing the acceleration to apply to the 
                respective car in self.cars
            time_step: number of seconds occuring during this update
            cycle: specifies whether to place cars in the beginning after
                they have crossed the finish line
        '''
        # Apply to Each Car
        for index in range(len(self.cars)):
            self.cars[index].update_position(accel_list[index], time_step)

        # Check Wall Crash
        for car in self.cars:
            if car.hit_wall():
                return -1

        # Check Car Crash
        car_spots = []
        for car in self.cars:
            new_back, new_front = car.position, car.position + CAR_LENGTH

            # Check for Any Crashes
            for spot in car_spots:
                back, front = spot[0], spot[1]
                if (back <= new_back and new_back <= front) or (back <= new_front and new_front <= front):
                    return -1

            # Add to List to Check
            car_spots.append([new_back, new_front])

        # Resolving Cars that Finished
        # currently displaces finished cars an arbitrary far distance away
        finished_count = 0
        for i, car in enumerate(self.cars):
            if self.finished_cars[i]:
                car.position = self.length*(i+2)*2
                finished_count += 1
            elif car.position >= self.length:
                car.position = self.length*(i+2)*2
                self.finished_cars[i] = True
                return 1
        if finished_count == len(self.cars):
            return 2
        
        return 0