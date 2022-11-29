import numbers

# from Configuration import StaticObstaclesConfiguration
import pygame
from typing import Union, List, Tuple, Any, Optional
from enum import Enum, auto

# Colors (defining colors for pygame)

nodeRad = 2  # radius of nodes on screen
nodeThickness = 0
edgeThickness = 1  # thickness of edge or line between nodes when displayed


class Color(Enum):
    grey = 1
    blue = 2
    green = 3
    red = 4
    white = 5
    purple = 6
    black = 7

    def value(self) -> Tuple[int, int, int]:
        if self == Color.grey:
            return (70, 70, 70)
        if self == Color.blue:
            return (0, 0, 255)
        if self == Color.green:
            return (0, 255, 0)
        if self == Color.red:
            return (255, 0, 0)
        if self == Color.white:
            return (255, 255, 255)
        if self == Color.purple:
            return (255, 0, 255)
        if self == Color.black:
            return 0, 0, 0
        else:
            return 0, 0, 0


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

    display_map.fill(Color.black.value())

    pygame.draw.circle(display_map, Color.green.value(), agent, nodeRad + 10, 0)
    pygame.draw.circle(display_map, Color.red.value(), goal, nodeRad + 15, 0)

    for obstacle in obstacles:
        pygame.draw.rect(display_map, Color.blue.value(), obstacle)

    pygame.display.update()
    pass


def draw_graph_and_path(display_map, vertices: List["StaticObstaclesConfiguration"],
                        path: Optional[List[Tuple["StaticObstaclesConfiguration", Any]]], v_color: Optional[Color] = None):
    """
    Draws a graph and a path on that graph
    :param vertices:
    :param path:
    :return:
    """
    v_color = Color.blue if v_color is None else v_color
    for v in vertices:
        pygame.draw.circle(display_map, v_color.value(), v.agent, 2, 0)

        if v.parent_vector is not None:
            # Draw the parent vector
            pygame.draw.line(display_map, v_color.value(), v.agent, v.parent_vector[0].agent, 1)

    if path is not None:
        for p in path:
            pygame.draw.circle(display_map, Color.red.value(), p[0].agent, 2, 0)

    pygame.display.update()
