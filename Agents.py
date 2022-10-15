from abc import ABC, abstractmethod
from Configuration import Configuration, Action
import pygame


class Agent(ABC):
    """
    An agent to play our Game
    """

    @abstractmethod
    def get_action(self, conf: Configuration, display_map: pygame.Surface = None) -> Action:
        """
        Gives the agent a configuration and the agent returns an action to perform
        :param display_map: A map to visualize decision making on, else None
        :param conf:
        :return:
        """