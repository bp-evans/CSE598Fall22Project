from Configuration import StaticObstaclesConfiguration


class RRT_Core:
    def __init__(self, seed_tree):
        self.G = seed_tree # Start off our RRT tree with the seed tree, I'm leaving this data structure undefined for now,
        # maybe we would like it to be a heap? or maybe a just a flat list? unsure which would be most performant right now,
        # or maybe it's just a Conf with children, but currently Conf doesn't store children so...
        pass

    def RRTAlg(self, k: int, dmax=20):  # -> Graph/Tree
        for i in range(k):
            # TODO: Change to extract Configuration type from self.G
            # For example, type = type(self.G.anyNode)
            conf_type = StaticObstaclesConfiguration # PLACEHOLDER

            n_rand = conf_type.gen_random_conf()  # This will be a valid random conf

            n_near = n_rand.nearest_vertex(self.G) # Make sure the input type agrees, change Configuration if you want

            n_next = n_rand.new_conf_from(n_near, dmax)

            self.G.add_node(n_next)
            # Maybe also add edge, but this could be redundant if the Conf already stores its parents


            # if (self.check_valid_state(n_rand)):
            #     n_near = static_config.nearest_vertex(n_rand)
            #     n_next = static_config.new_conf_from(n_rand, n_near, dmax)
            #     self.G.add_vertex(n_next)
            #     self.G.add_edge(n_next)

        return self.G  # Some layer above this will visualize the final graph


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
