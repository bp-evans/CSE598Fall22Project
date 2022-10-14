import numpy as np
from abc import ABC, abstractmethod

"""
Use `Configuration` in all code references, but instantiate `StaticObstaclesConfiguration` to begin with.
"""


class Configuration(ABC):
    """
    Abstract class that represents a game configuration. Implement child classes of this for specific games.
    Intuition for these methods comes from: https://en.wikipedia.org/wiki/Rapidly-exploring_random_tree
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_vector(self) -> np.ndarray:
        """
        Returns this configuration object encoded as a 1-D array
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


class StaticObstaclesConfiguration(Configuration):
    """
    Configuration for a game where the obstacles are static.

    TODO: Implement this!
    """

    def __init__(self, agent: (float, float), goal: (float, float)):
        """

        :param agent:
        :param goal:
        """
        # The state is going to be the (x,y) coord of the agent and the goal
        self.state = np.array(list(agent) + list(goal))
        self.parent = None
        pass

    def get_vector(self) -> np.ndarray:
        pass

    @staticmethod
    def gen_random_conf() -> "StaticObstaclesConfiguration":
        pass

    def nearest_vertex(self, vertices: "StaticObstaclesConfiguration") -> "StaticObstaclesConfiguration":
        pass

    def new_conf_from(self, near: "StaticObstaclesConfiguration", delta) -> "StaticObstaclesConfiguration":
        pass

    def take_action(self, action) -> "Configuration":
        pass
