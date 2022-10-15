# keyboardAgents.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from Agents import Agent
from Configuration import Configuration, DiscreteDirectionAction
import random
import pygame
from typing import Union, Optional


class KeyboardAgent(Agent):
    """
    An agent controlled by the keyboard.
    """

    def __init__(self, index=0):

        self.lastMove = DiscreteDirectionAction.STOP
        # self.index = index
        # self.keys = []

    def get_action(self, conf: Configuration, display_map: pygame.Surface = None) -> DiscreteDirectionAction:
        # from graphicsUtils import keys_waiting
        # from graphicsUtils import keys_pressed
        # keys = list(keys_waiting()) + list(keys_pressed())
        # if keys != []:
        #     self.keys = keys

        legal = conf.get_legal_actions()
        move = self.getMove(legal)

        # if move == DiscreteDirectionAction.STOP:
        #     # Try to move in the same direction as before
        #     if self.lastMove in legal:
        #         move = self.lastMove

        # if (self.STOP_KEY in self.keys) and DiscreteDirectionAction.STOP in legal: move = DiscreteDirectionAction.STOP

        if move not in legal:
            move = random.choice(legal)

        self.lastMove = move
        return move

    def getMove(self, legal):
        # move = DiscreteDirectionAction.STOP
        # Check for keys pressed
        key_pressed = False

        def check_keys(keys) -> Optional[DiscreteDirectionAction]:
            if (pygame.K_w in keys) and DiscreteDirectionAction.NORTH in legal:
                return DiscreteDirectionAction.NORTH
            elif (pygame.K_d in keys) and DiscreteDirectionAction.EAST in legal:
                return DiscreteDirectionAction.EAST
            elif (pygame.K_s in keys) and DiscreteDirectionAction.SOUTH in legal:
                return DiscreteDirectionAction.SOUTH
            elif (pygame.K_a in keys) and DiscreteDirectionAction.WEST in legal:
                return DiscreteDirectionAction.WEST

        all_keys = pygame.key.get_pressed()
        pressed = [pygame.K_w] if all_keys[pygame.K_w] else []
        pressed += [pygame.K_d] if all_keys[pygame.K_d] else []
        pressed += [pygame.K_s] if all_keys[pygame.K_s] else []
        pressed += [pygame.K_a] if all_keys[pygame.K_a] else []
        action = check_keys(pressed)

        while action is None:
            keys = []
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    keys.append(event.key)

            action = check_keys(keys)

        pygame.event.pump()

        # if (self.WEST_KEY in self.keys or 'Left' in self.keys) and DiscreteDirectionAction.WEST in legal:  move = DiscreteDirectionAction.WEST
        # if (self.EAST_KEY in self.keys or 'Right' in self.keys) and DiscreteDirectionAction.EAST in legal: move = DiscreteDirectionAction.EAST
        # if (self.NORTH_KEY in self.keys or 'Up' in self.keys) and DiscreteDirectionAction.NORTH in legal:   move = DiscreteDirectionAction.NORTH
        # if (self.SOUTH_KEY in self.keys or 'Down' in self.keys) and DiscreteDirectionAction.SOUTH in legal: move = DiscreteDirectionAction.SOUTH
        return action


# class KeyboardAgent2(KeyboardAgent):
#     """
#     A second agent controlled by the keyboard.
#     """
#     # NOTE: Arrow keys also work.
#     WEST_KEY = 'j'
#     EAST_KEY = "l"
#     NORTH_KEY = 'i'
#     SOUTH_KEY = 'k'
#     STOP_KEY = 'u'
#
#     def getMove(self, legal):
#         move = Directions.STOP
#         if (self.WEST_KEY in self.keys) and Directions.WEST in legal:  move = Directions.WEST
#         if (self.EAST_KEY in self.keys) and Directions.EAST in legal: move = Directions.EAST
#         if (self.NORTH_KEY in self.keys) and Directions.NORTH in legal:   move = Directions.NORTH
#         if (self.SOUTH_KEY in self.keys) and Directions.SOUTH in legal: move = Directions.SOUTH
#         return move
