import math

import numpy as np
from abc import ABC, abstractmethod
from enum import Enum

import pygame

from typing import List, Optional, Tuple
from Visualizers import visualize_conf

"""
Use `Configuration` in all code references, but instantiate `StaticObstaclesConfiguration` to begin with.
"""


class Action:
    """
    Represents an action (either continuous or discrete)
    """
    pass


class Configuration(ABC):
    """
    Abstract class that represents a game configuration. Implement child classes of this for specific games.
    Intuition for these methods comes from: https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def as_vector(self) -> np.ndarray:
        """
        Returns this configuration object encoded as a 1-D array
        :return:
        """
        pass

    @abstractmethod
    def get_parent_vector(self) -> ("Configuration", Action):
        """
        Every configuration is the result of some action taken on a parent conf, this returns the parent conf and action taken
        to get this conf
        :return:
        """

    @abstractmethod
    def get_legal_actions(self) -> [Action]:
        """
        Return the legal actions that can be taken from this state
        :return:
        """
        pass

    @staticmethod
    @abstractmethod
    def gen_random_conf() -> "Configuration":
        """
        Generates a random _valid_ configuration
        :return:
        """
        pass

    @abstractmethod
    def nearest_vertex(self, vertices: "Configuration") -> "Configuration":
        """
        Returns the vertex in the list of vertices that is closest to this one.
        :param vertices:
        :return:
        """
        pass

    @abstractmethod
    def new_conf_from(self, near: "Configuration", delta) -> "Configuration":
        """
        Returns a new configuration object that is delta away from "near" conf _towards_ this conf.
        Would also probably need to store the parent and action associated to get this.
        :param near:
        :param delta:
        :return:
        """
        pass

    @abstractmethod
    def take_action(self, action) -> "Configuration":
        """
        Returns a new configuration that is obtained from taking the specified action from this conf.
        :param action:
        :return:
        """
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        """
        Checks if this conf space is terminal, i.e. has reached goal
        :return:
        """
        pass

    @abstractmethod
    def visualize(self, display_map):
        """
        Visualize this configuration somehow, if possible
        :param display_map: The map to visualize onto
        :return:
        """
        pass


class DiscreteDirectionAction(Action, Enum):
    NORTH = 1
    SOUTH = 3
    EAST = 2
    WEST = 4
    STOP = 0

    def direction_vector(self) -> np.ndarray:
        if self == DiscreteDirectionAction.NORTH:
            return np.array([0, -1])
        elif self == DiscreteDirectionAction.SOUTH:
            return np.array([0, 1])
        elif self == DiscreteDirectionAction.EAST:
            return np.array([1, 0])
        elif self == DiscreteDirectionAction.WEST:
            return np.array([-1, 0])
        else:
            return np.array([0, 0])


class StaticObstaclesConfiguration(Configuration):
    """
    Configuration for a game where the obstacles are static.

    TODO: Implement this!
    """

    mapw = 800
    maph = 500

    @staticmethod
    def obstacles() -> ["Obstacles"]:
        """
        Stores the obstacles, returns a list of the obstacles
        :return:
        """
        return [(100, 0, 30, 200),
                (50, 350, 30, 30),
                (250, 300, 30, 200),
                (250, 0, 30, 150),
                (450, 0, 30, 400),
                (550, 200, 30, 30),
                (575, 400, 30, 30),
                (650, 300, 30, 30),
                (650, 70, 30, 30)
                ]

    def __init__(self, agent: (float, float), goal: (float, float)):
        """

        :param agent:
        :param goal:
        """
        # The state is going to be the (x,y) coord of the agent and the goal
        self.agent = np.array(list(agent))
        self.goal = np.array(list(goal))
        self.parent_vector = None
        # passing in obstacles from the abstract runner
        pass

    def get_legal_actions(self) -> [DiscreteDirectionAction]:
        # Let all actions be valid
        return [DiscreteDirectionAction(i) for i in range(1, 5)]

    def is_terminal(self) -> bool:
        # Is terminal if the agent is close enough to the goal
        return np.linalg.norm(self.goal - self.agent) < 30

    def get_parent_vector(self) -> Optional[Tuple["StaticObstaclesConfiguration", DiscreteDirectionAction]]:
        return self.parent_vector

    def as_vector(self) -> np.ndarray:
        # Return this state encoded as a vector
        return np.append(self.agent, self.goal)

    @staticmethod
    def gen_random_conf() -> "StaticObstaclesConfiguration":
        # TODO: Implement, This should only return a VALID conf

        is_valid = False
        while (not is_valid):
            x = np.random.uniform(0, StaticObstaclesConfiguration.mapw)
            y = np.random.uniform(0, StaticObstaclesConfiguration.maph)
            # x = int(random.uniform(0, StaticObstaclesConfiguration.mapw))
            # y = int(random.uniform(0, StaticObstaclesConfiguration.maph))
            is_valid = True
            # Check if this new conf collides with any obstacles
            for obstacle in StaticObstaclesConfiguration.obstacles():
                if pygame.Rect(obstacle).collidepoint(x, y):
                    is_valid = False

        ret_config = StaticObstaclesConfiguration((x, y), (0, 0))  # Goal is weird, but doesn't really matter
        return ret_config

    def nearest_vertex(self, vertices: List["StaticObstaclesConfiguration"]) -> "StaticObstaclesConfiguration":
        dmin = 999
        near = None
        for vertex in vertices:
            dist = np.linalg.norm(self.agent - vertex.agent)
            if dist < dmin:
                near = vertex
                dmin = dist
        return near

    def new_conf_from(self, near: "StaticObstaclesConfiguration") -> Optional["StaticObstaclesConfiguration"]:
        # Should only return a VALID conf where the parent, and action are set on this conf already
        diff = self.agent - near.agent
        theta = math.atan2(diff[1], diff[0])

        if 3/4 * math.pi >= theta >= math.pi / 4:
            # Move south
            action = DiscreteDirectionAction.SOUTH
        elif math.pi / 4 >= theta >= -1 * math.pi / 4:
            # Move east
            action = DiscreteDirectionAction.EAST
        elif -1 * math.pi / 4 >= theta >= -3/4 * math.pi:
            # Move north
            action = DiscreteDirectionAction.NORTH
        else:
            # Move west
            action = DiscreteDirectionAction.WEST

        # Can either change step size to be static or variable
        step = 20  # min(50, min(np.abs(diff)))
        new = near.take_action(action, step)
        # Set the parent so that we can retrace later
        new.parent_vector = (near, action)

        # Check to make sure this doesn't cross an obstacle
        diff = new.agent - near.agent
        for o in StaticObstaclesConfiguration.obstacles():
            obstacle = pygame.Rect(o)
            prev = None
            for i in range(101):
                u = i / 100
                test_point = diff * u + near.agent
                if obstacle.collidepoint(test_point):
                    # They collide, don't expand this way
                    return None
                    # if prev is not None:
                    #     new.agent = prev
                    # else:
                    #     return None
                else:
                    prev = test_point

        return new

    def take_action(self, action: DiscreteDirectionAction, step_dist = 5) -> "StaticObstaclesConfiguration":
        # Returns the conf that results from taking an action
        new_conf = StaticObstaclesConfiguration(self.agent + (step_dist * action.direction_vector()), self.goal)
        return new_conf

    def visualize(self, display_map):
        visualize_conf(display_map, self.agent, self.goal, self.obstacles())
        pass
