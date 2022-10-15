from Agents import Agent
from RRTAbstract import RRT_Core
from Configuration import Configuration, Action


class RRTAgent(Agent):
    """
    This agent gets the next action by building an RRT tree from scratch and then using the found path
    """

    def get_action(self, conf: Configuration) -> Action:
        rrt = RRT_Core([conf])
        graph, path = rrt.RRTAlg(1000)

        return path[0][1]


class BCRRTAgent(Agent):
    """
    This agent gets the next action by building an RRT tree seeded with the rollout of a BC policy
    """
    def __init__(self, policy):
        self.policy = policy

    def get_action(self, conf: Configuration) -> Action:
        # TODO: Rollout BC policy,

        seed_tree = self.policy.rollout(conf)
        rrt = RRT_Core(seed_tree)
        graph, path = rrt.RRTAlg(1000)

        return path[0][1]