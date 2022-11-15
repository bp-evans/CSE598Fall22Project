import time

import torch
from abc import abstractmethod
import Visualizers
from Agents import Agent
import numpy as np
from RRTAbstract import RRT_Core
from RRTAgent import Observer
from BehaviorCloning import MLP
from Configuration import DiscreteDirectionAction, Configuration, Action
import pygame
import random


class ModelAgent(Agent):
    """
    This agent gets the next action by building an RRT tree seeded with the rollout of a BC policy
    """

    def __init__(self):
        pass

    def get_action(self, conf: Configuration, display_map: pygame.Surface = None) -> Action:
        seed_tree = self.rollout(conf, 5)  # 5 rollouts
        if display_map is not None:
            # Show the rolled-out tree
            Visualizers.draw_graph_and_path(display_map, seed_tree, None, v_color=Visualizers.Color.purple)
            time.sleep(0.01)
        rrt = RRT_Core(seed_tree)
        graph, path = rrt.RRTAlg(1000, Observer(display_map, conf, first_highlighted=len(seed_tree)))

        if path is None:
            closest = graph[0]
            min_dist = closest.dist_to_terminal()
            for v in graph:
                dist = v.dist_to_terminal()
                if dist < min_dist:
                    closest = v
                    min_dist = dist

            next_step = closest
            last_vec = None
            while next_step.get_parent_vector() is not None:
                last_vec = next_step.get_parent_vector()
                next_step = next_step.get_parent_vector()[0]
            if last_vec is None:
                print("Error: RRT tree found that current conf is closest to goal. Taking random action.")
                return random.choice(conf.get_legal_actions())
            return last_vec[1]
        else:
            return path[0][1]

    def rollout(self, conf: Configuration, rollout_episodes: int):
        """
        Rolls out the BC policy for N iters starting from conf and builds a tree, then runs RRT on that tree
        :param rollout_episodes:
        :return:
        """
        G = [conf]
        while rollout_episodes:
            roll_out_depth = 20
            while roll_out_depth:
                action = self.get_model_output(conf)
                new_conf = conf.take_action(action)

                roll_out_depth -= 1
                if not new_conf.is_valid_conf():
                    continue
                new_conf.parent_vector = (conf, action)
                G.append(new_conf)
                conf = new_conf
            rollout_episodes -= 1

        return G

    @abstractmethod
    def get_model_output(self, conf: Configuration) -> Action:
        """
        Implement this to return the output action from a model
        :param surface:
        :param conf:
        :return:
        """
        pass


class BCRRTAgent(ModelAgent):
    def __init__(self, bc_model: str):
        saved_states = torch.load(bc_model)
        self.model = MLP(input_shape=4)
        self.model.load_state_dict(saved_states['state_dict'])
        self.model.eval()
        super(BCRRTAgent, self).__init__()

    def get_model_output(self, conf: Configuration) -> Action:
        x = torch.tensor(conf.as_vector(), dtype=torch.float, requires_grad=False)
        x = x.view(1, 4)
        outputs = self.model(x)
        _, predicted = torch.max(outputs.data, 1)
        predicted = int(predicted)
        predicted += 1  # index start from 0
        action = DiscreteDirectionAction(predicted)
        return action


class ImageBCRRRTAgent(ModelAgent):
    def __init__(self, jit_path: str):
        self.model = torch.jit.load(jit_path, map_location=torch.device("cpu"))
        self.model.eval()
        super(ImageBCRRRTAgent, self).__init__()

    def get_model_output(self, conf: Configuration) -> Action:
        # Need to visualize the conf on a background surface
        surface = pygame.Surface((conf.mapw, conf.maph))
        conf.visualize(surface)

        # Get the surface as an image
        surf = pygame.surfarray.pixels3d(surface)
        surf = surf.swapaxes(0,2)
        x = torch.from_numpy(surf.copy())
        outputs = self.model(x)
        predicted = torch.argmax(outputs.data)
        predicted = int(predicted)
        action = DiscreteDirectionAction(predicted)
        return action
