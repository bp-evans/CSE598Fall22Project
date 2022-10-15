import numpy as np
import pygame
from typing import Union


# Colors (defining colors for pygame)
grey = (70, 70, 70)
blue = (0, 0, 255)
green = (0, 255, 0)
red = (255, 0, 0)
white = (255, 255, 255)
nodeRad = 2 # radius of nodes on screen
nodeThickness = 0
edgeThickness = 1 # thickness of edge or line between nodes when displayed


def visualize_conf(display_map: Union[pygame.Surface,pygame.SurfaceType], agent: (float, float), goal: (float, float),
                   obstacles: [(float, float, float, float)]):
    """
    Visualize these objects on the specified map
    :param display_map:
    :param agent:
    :param goal:
    :param obstacles:
    :return:
    """

    display_map.fill((255, 255, 255))

    print("Drawing objects")

    pygame.draw.circle(display_map, green, agent, nodeRad + 5, 0)
    pygame.draw.circle(display_map, green, goal, nodeRad + 20, 1)

    for obstacle in obstacles:
        pygame.draw.rect(display_map, grey, obstacle)

    pygame.display.update()
    pass
