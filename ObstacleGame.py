import time
import uuid
import os
import csv
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

    def play(self, isDynamic, visual=True, dyn_start=False, dyn_goal=False):
        """
        Plays the game.
        Starts a game loop where every iteration the agent is queried for an action. The action that
        the agent takes updates the game state in some way. Terminal condition is checked, and then it loops.

        :return:
        """

        # Begin with some configuration
        if(isDynamic):
            print("Dynamic Obstacles")
            _conft = DynamicObstaclesConfiguration
        else:
            print("Static Obstacles")
            _conft = StaticObstaclesConfiguration

        conf = _conft.gen_random_conf(set_start=None if dyn_start else (50, 50), set_goal=None if dyn_goal else (700, 200))
        # Grab current demonstration labels
        demonstration_label_file = "ImageLabels.csv"
        images_dir = "image_demos/"
        exists = os.path.isfile(demonstration_label_file)
        demonstration_labels = open(demonstration_label_file, 'a')
        label_writer = csv.DictWriter(demonstration_labels, ["Image Name", "Label"])
        if not exists:
            label_writer.writeheader()

        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        # Game Loop
        # demonstration = list()
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
                # Switch over to using images as demos
                # Write an image of the game space to image_demos
                filename = str(uuid.uuid1()) + ".jpg"
                pygame.image.save(self.display_map, images_dir+filename)
                # Save this label
                new_label = {"Image Name": filename, "Label": action.value}
                label_writer.writerow(new_label)

                # demonstration.append((conf, action))
            # Take that action and update the conf
            print("Updating game based on action: " + str(action))
            new_conf = conf.take_action(action)
            conf = new_conf if new_conf.is_valid_conf() else conf

            time.sleep(0.05)

        # Show the result,
        print("Game Ended")

        # Save demonstrations
        demonstration_labels.close()