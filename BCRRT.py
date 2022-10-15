
class BCRRT:
    """
    This is essentially Max's Algorithm 1 from "Efficient Exploration via First-Person Behavior Cloning Assisted Rapidly-Exploring Random Trees"
    """
    def __init__(self, conf, policy):
        """
        Takes in a BC policy and a start conf
        :param policy:
        """
        pass

    def run(self, N: int):
        """
        Rolls out the BC policy for N iters starting from conf and builds a tree, then runs RRT on that tree
        :param N:
        :return:
        """
        # rollout policy from conf
        # run RRT with the seed_tree generated
        # Return tree/Graph
