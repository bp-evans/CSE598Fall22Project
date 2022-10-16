import time

import pygame
import random

from Agents import Agent
from Configuration import StaticObstaclesConfiguration, DynamicObstaclesConfiguration


class ObstacleGame:
    """
    This class manages running 1 of our Obstacles Game.
    It can be run either with visuals (pygame) or without.

    To play it, you need to give it an agent.
    """

    def __init__(self, agent: Agent, demo=False):
        pygame.init()
        # Draw my static game space

        self.agent = agent
        self.demo = demo

        dimensions = (500, 800)
        maph, mapw = dimensions

        StaticObstaclesConfiguration.mapw = mapw
        StaticObstaclesConfiguration.maph = maph

        self.display_map = pygame.display.set_mode((mapw, maph))  # setting up pygame window of proper dimensions

    @classmethod
    def get_start_goal_preset(cls):
        return StaticObstaclesConfiguration((50, 50), (800, 300))

    def play(self, visual=True):
        """
        Plays the game.
        Starts a game loop where every iteration the agent is queried for an action. The action that
        the agent takes updates the game state in some way. Terminal condition is checked, and then it loops.

        :return:
        """

        # Begin with some configuration
        conf = StaticObstaclesConfiguration((50, 50), (800, 300))

        # Game Loop
        demonstration = list()
        while not conf.is_terminal():
            # If requested, visualize the game conf
            if visual:
                print("Visualizing game state")
                conf.visualize(self.display_map)
                pygame.event.pump()

            # Query the agent for an action
            print("Requesting action")
            action = self.agent.get_action(conf, self.display_map if visual else None)

            if self.demo:
                demonstration.append((conf, action))
            # Take that action and update the conf
            print("Updating game based on action: " + str(action))
            conf = conf.take_action(action)

            time.sleep(1)

        # Show the result,
        print("Game Ended")

        return demonstration


class DynamicObstacleGame:
    """
    This class manages running 1 of our Obstacles Game.
    It can be run either with visuals (pygame) or without.

    To play it, you need to give it an agent.
    """

    def __init__(self, agent: Agent, demo=False):
        pygame.init()
        # Draw my static game space

        self.agent = agent
        self.demo = demo

        dimensions = (500, 800)
        maph, mapw = dimensions

        DynamicObstaclesConfiguration.mapw = mapw
        DynamicObstaclesConfiguration.maph = maph

        self.display_map = pygame.display.set_mode((mapw, maph))  # setting up pygame window of proper dimensions

    def play(self, visual=True):
        """
        Plays the game.
        Starts a game loop where every iteration the agent is queried for an action. The action that
        the agent takes updates the game state in some way. Terminal condition is checked, and then it loops.

        :return:
        """

        # Begin with some configuration
        start = (random.randint(1, 79) * 10, random.randint(1, 49) * 10)
        goal = (random.randint(1, 79) * 10, random.randint(1, 49) * 10)
        while pygame.Rect(start[0], start[1], 100, 100).collidepoint(goal):
            goal = (random.randint(1, 79) * 10, random.randint(1, 49) * 10)

        conf = DynamicObstaclesConfiguration(start, goal)

        demonstration = list()
        # Game Loop
        while not conf.is_terminal():
            # If requested, visualize the game conf
            if visual:
                print("Visualizing game state")
                conf.visualize(self.display_map)

            # Query the agent for an action
            print("Requesting action")
            action = self.agent.get_action(conf)

            if self.demo:
                demonstration.append((conf, action))

            # Take that action and update the conf
            print("Updating game based on action: " + str(action))
            conf = conf.take_action(action)

            time.sleep(0.05)
            pass

        # Show the result,
        print("Game Ended")
        return demonstration
