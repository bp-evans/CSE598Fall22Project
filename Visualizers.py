import numbers

# from Configuration import StaticObstaclesConfiguration
import pygame
from typing import Union, List, Tuple, Any, Optional

# Colors (defining colors for pygame)
grey = (70, 70, 70)
blue = (0, 0, 255)
green = (0, 255, 0)
red = (255, 0, 0)
white = (255, 255, 255)
purple = (255, 0, 255)
nodeRad = 2  # radius of nodes on screen
nodeThickness = 0
edgeThickness = 1  # thickness of edge or line between nodes when displayed


def visualize_conf(display_map: Union[pygame.Surface, pygame.SurfaceType], agent: (float, float), goal: (float, float),
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

    # print("Drawing objects")

    pygame.draw.circle(display_map, green, agent, nodeRad + 5, 0)
    pygame.draw.circle(display_map, green, goal, nodeRad + 20, 1)

    for obstacle in obstacles:
        pygame.draw.rect(display_map, grey, obstacle)

    pygame.display.update()
    pass


def draw_graph_and_path(display_map, vertices: List["StaticObstaclesConfiguration"],
                        path: Optional[List[Tuple["StaticObstaclesConfiguration", Any]]]):
    """
    Draws a graph and a path on that graph
    :param vertices:
    :param path:
    :return:
    """
    for v in vertices:
        pygame.draw.circle(display_map, blue, v.agent, 2, 0)

        if v.parent_vector is not None:
            # Draw the parent vector
            pygame.draw.line(display_map, blue, v.agent, v.parent_vector[0].agent, 1)

    if path is not None:
        for p in path:
            pygame.draw.circle(display_map, red, p[0].agent, 2, 0)

    pygame.display.update()
