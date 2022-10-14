from Configuration import Configuration, StaticObstaclesConfiguration
from abc import ABC, abstractmethod

class RRT_Core:
    def __init__(self, start, goal, obstacles):
        self.G = []
        self.obs = obstacles # array of obstacles
        config = Configuration()
        static_config = StaticObstaclesConfiguration(ABC)
        pass

    def check_value_state(self, nrand): # make sure the random_conf isn't on an obstacle
        pass

    def RRTAlg(self, dmax = 20):
        n_rand = static_config.get_random_conf()
        if(self.check_valid_state(n_rand)):
            n_near = static_config.nearest_vertex(n_rand)
            n_next = static_config.new_conf_from(n_rand, n_near, dmax)
            self.G.add_vertex(n_next)
            self.G.add_edge(n_next)
        

