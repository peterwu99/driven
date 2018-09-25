import numpy as np
import os
import pygame
from pygame import rect, draw
from pygame.locals import *

from constants import ENVS
from env import Environment
from control import get_accel_list, load_policy

METERS_TO_PIXELS = 20
sMap = ENVS["FOUR_WAY_1"]

# Sprite Classes
class CarSprite(object):
    def __init__(self, car):
        self.car = car

    def draw(self, screen):
        points = [(x * METERS_TO_PIXELS, y * METERS_TO_PIXELS) for x, y in self.car.points()]
        draw.polygon(screen, (128, 0, 0), points)


class EnvSprite(object):
    def __init__(self, env):
        self.env = env
        self.car_sprites = [CarSprite(car) for car in env.cars]

    def draw(self, screen):
        screen.fill((255, 255, 255))

        # Draw Grid
        for i in range(len(sMap)):
            for j in range(len(sMap[i])):
                if sMap[i][j] == 0:
                    continue

                draw.rect(screen, (105, 105, 105), rect.Rect(
                    i * METERS_TO_PIXELS,
                    j * METERS_TO_PIXELS,
                    METERS_TO_PIXELS,
                    METERS_TO_PIXELS
                ))

        # Draw Cars
        for car in self.car_sprites:
            car.draw(screen)


# Initialize the Environment
env = Environment()

# Pygame Rendering
os.environ["SDL_VIDEO_CENTERED"] = "1"
pygame.init()

mainScreen = pygame.display.set_mode((len(sMap) * METERS_TO_PIXELS, len(sMap[0]) * METERS_TO_PIXELS))
pygame.display.set_caption("Running Simulation")
clock = pygame.time.Clock()
env_sprite = EnvSprite(env)

# Get Policy
policy = load_policy()

# Main Loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()

    test = env_sprite.env.update(get_accel_list(policy, env_sprite.env.cars))

    # Draw Environment on Screen
    env_sprite.draw(mainScreen)
    pygame.display.update()

    # Progress forward
    clock.tick(15)
