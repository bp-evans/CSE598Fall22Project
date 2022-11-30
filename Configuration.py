import math
import numbers
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
import pygame
import random
from typing import List, Optional, Tuple, Iterable
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
        Generates a completely random _valid_ configuration
        :return:
        """
        pass

    def randomized_agent(self) -> "Configuration":
        """
        Returns a configuration the same as self but with just the agent's state randomized
        :return:
        """

    @abstractmethod
    def nearest_vertex(self, vertices: List["Configuration"]) -> "Configuration":
        """
        Returns the vertex in the list of vertices that is closest to this one.
        :param vertices:
        :return:
        """
        pass

    @abstractmethod
    def new_conf_from(self, near: "Configuration") -> "Configuration":
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
    def dist_to_terminal(self) -> numbers.Real:
        """
        Returns the distance from this conf to a terminal conf.
        :return:
        """

    @abstractmethod
    def visualize(self, display_map):
        """
        Visualize this configuration somehow, if possible
        :param display_map: The map to visualize onto
        :return:
        """
        pass


class DiscreteDirectionAction(Action, Enum):
    NORTH = 0
    SOUTH = 2
    EAST = 1
    WEST = 3
    STOP = -1

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
    """
    mapw = 800
    maph = 500

    @staticmethod
    def gen_obstacles(within: Tuple[int, int], valid_fn) -> List["Obstacle"]:
        """
        Generate an obstacle list when requested
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

    def __init__(self, agent: Iterable[float], goal: Iterable[float], obstacles: Optional[List["Obstacles"]] = None):
        """

        :param agent:
        :param goal:
        """
        # The state is going to be the (x,y) coord of the agent and the goal
        self.agent = np.array(list(agent))
        self.goal = np.array(list(goal))
        self.parent_vector = None
        self.max_obs = 15

        def valid(obs):
            self.obstacle_list = obs
            return self.is_valid_conf()
        # Get valid obstacles
        self.obstacle_list = self.gen_obstacles(
            within=(StaticObstaclesConfiguration.mapw, StaticObstaclesConfiguration.maph),
            valid_fn=valid) if obstacles is None else obstacles

    def is_valid_conf(self) -> bool:
        """
        Given a conf validate if it is valid in current environment.
        """
        for obs in self.obstacle_list:
            if pygame.Rect(obs).collidepoint(list(self.agent)) or pygame.Rect(obs).collidepoint(list(self.goal)):
                return False
        return True

    def get_legal_actions(self) -> [DiscreteDirectionAction]:
        return list(
            filter(lambda c: self.take_action(c).is_valid_conf(), [DiscreteDirectionAction(i) for i in range(0, 4)]))

    def is_terminal(self) -> bool:
        # Is terminal if the agent is close enough to the goal
        return np.linalg.norm(self.goal - self.agent) < 30

    def dist_to_terminal(self) -> numbers.Real:
        return np.linalg.norm(self.goal - self.agent)

    def get_parent_vector(self) -> Optional[Tuple["StaticObstaclesConfiguration", DiscreteDirectionAction]]:
        return self.parent_vector

    def as_vector(self) -> np.ndarray:
        # Return this state encoded as a vector
        return np.append(self.agent, self.goal)

    @classmethod
    def gen_random_conf(cls, set_start:Optional[Iterable[float]] = None, set_goal: Optional[Iterable[float]] = None,
                        set_obstacles: Optional[List["Obstacle"]] = None) -> "StaticObstaclesConfiguration":
        # This should only return a VALID conf

        valid = False
        while not valid:
            if set_start is None:
                x = np.random.uniform(0, StaticObstaclesConfiguration.mapw)
                y = np.random.uniform(0, StaticObstaclesConfiguration.maph)
                start = (x,y)
            else:
                start = set_start
            if set_goal is None:
                goal = [0, 0]
                goal[0] = np.random.uniform(0,StaticObstaclesConfiguration.mapw)
                goal[1] = np.random.uniform(0, StaticObstaclesConfiguration.maph)
            else:
                goal = set_goal

            ret_config = cls(start, goal, set_obstacles)  # Goal is weird, but doesn't really matter
            valid = ret_config.is_valid_conf()
        return ret_config

    def randomized_agent(self) -> "StaticObstaclesConfiguration":
        return self.gen_random_conf(set_goal=self.goal, set_obstacles=self.obstacle_list)

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

        if 3 / 4 * math.pi >= theta >= math.pi / 4:
            # Move south
            action = DiscreteDirectionAction.SOUTH
        elif math.pi / 4 >= theta >= -1 * math.pi / 4:
            # Move east
            action = DiscreteDirectionAction.EAST
        elif -1 * math.pi / 4 >= theta >= -3 / 4 * math.pi:
            # Move north
            action = DiscreteDirectionAction.NORTH
        else:
            # Move west
            action = DiscreteDirectionAction.WEST

        # Can either change step size to be static or variable
        # step = 20  # min(50, min(np.abs(diff)))
        new = near.take_action(action)
        # Set the parent so that we can retrace later
        new.parent_vector = (near, action)

        # Check to make sure this doesn't cross an obstacle
        diff = new.agent - near.agent
        for o in new.obstacle_list:
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

    def take_action(self, action: DiscreteDirectionAction, step_dist=10) -> "StaticObstaclesConfiguration":
        # Returns the conf that results from taking an action
        # Goal and obstacles remain the same
        new_conf = type(self)(self.agent + (step_dist * action.direction_vector()), self.goal, self.obstacle_list)
        return new_conf

    def visualize(self, display_map):
        visualize_conf(display_map, self.agent, self.goal, self.obstacle_list)
        pass

    def get_obs_vector(self):
        # TODO: This should probably move to as_vector() method in the DynamicObstaclesConfiguration
        return np.append(self.obstacle_list[0], self.obstacle_list[1:] +
                         [(0, 0, 0, 0) for _ in range(self.max_obs - len(self.obstacle_list))])


class DynamicObstaclesConfiguration(StaticObstaclesConfiguration):
    """
    Configuration for a game where the obstacles are static.
    """
    @staticmethod
    def gen_obstacles(within: (int, int), valid_fn) -> List["Obstacle"]:
        window = within

        obstacle_count = random.randint(6, 15)

        # rect (left, top, width, height)
        obstacle_list = list()
        while len(obstacle_list) < obstacle_count:
            top = random.randint(10, window[1] - 1)
            left = random.randint(10, window[0] - 1)
            height = random.randint(80, window[1])
            width = 30

            skip_this = False
            rect = pygame.Rect(left - 15, top - 15, width + 30, height + 30)

            if not valid_fn(obstacle_list + [(left, top, width, height)]):
                skip_this = True
            for l, t, w, h in obstacle_list:
                if rect.colliderect(pygame.Rect(l, t, w, h)):
                    skip_this = True
                    break
            if not skip_this:
                obstacle_list.append((left, top, width, height))
        return obstacle_list
