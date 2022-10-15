from typing import List

from Agents import Agent
from RRTAbstract import RRT_Core, RRTObserver
from Configuration import Configuration, Action
import random
import pygame
import Visualizers


class RRTAgent(Agent):
    """
    This agent gets the next action by building an RRT tree from scratch and then using the found path
    """

    def get_action(self, conf: Configuration, display_map: pygame.Surface = None) -> Action:
        class Observer(RRTObserver):
            def __init__(self, map):
                self.i = 0
                self.map = map

            def rrt_expanded(self, vertices: List[Configuration], rand: Configuration, near: Configuration,
                             nnext: Configuration):
                self.i += 1
                if self.map is not None and self.i % 10 == 0:
                    # print("Showing partial RRT tree")
                    conf.visualize(self.map)
                    Visualizers.draw_graph_and_path(self.map, vertices, None)
                    # Interact
                    pygame.draw.circle(self.map, Visualizers.red, rand.agent, 5, 0)
                    pygame.draw.circle(self.map, Visualizers.green, near.agent, 4, 0)
                    pygame.draw.circle(self.map, Visualizers.purple, nnext.agent, 3, 0) if nnext is not None else """"""
                    pygame.display.update()
                    pygame.event.wait(1)
                    # waiting = self.interactive
                    # while waiting:
                    #     for event in pygame.event.get():
                    #         if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                    #             waiting = False
                pass

            def rrt_terminated(self, found_terminal: bool):
                print(
                    "RRT algo terminated after " + str(self.i) + " expansions. Found terminal: " + str(found_terminal))

        rrt = RRT_Core([conf])
        graph, path = rrt.RRTAlg(1000, Observer(display_map))

        if path is None:
            # TODO: Check that this makes sense if a terminal node wasn't found in the tree
            # Get the vertex closest to the goal and go there
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


class BCRRTAgent(Agent):
    """
    This agent gets the next action by building an RRT tree seeded with the rollout of a BC policy
    """

    def __init__(self, policy):
        self.policy = policy

    def get_action(self, conf: Configuration, display_map: pygame.Surface = None) -> Action:
        # TODO: Rollout BC policy,

        seed_tree = self.policy.rollout(conf)
        rrt = RRT_Core(seed_tree)
        graph, path = rrt.RRTAlg(1000)

        return path[0][1]
