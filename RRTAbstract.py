from Configuration import Configuration, Action
from typing import List, Tuple, Optional
from abc import abstractmethod

class RRTObserver:
    """
    An observer object that can receive updates about an RRT algo's progression.
    """
    @abstractmethod
    def rrt_expanded(self, vertices: List[Configuration], rand: Configuration, near: Configuration, nnext: Configuration):
        pass

    @abstractmethod
    def rrt_terminated(self, found_terminal: bool):
        pass

class RRT_Core:
    """
    An RRT algorithm that is state agnostic.
    """
    def __init__(self, seed_tree):
        self.G: List[Configuration] = seed_tree
        pass

    def RRTAlg(self, k: int, observer: RRTObserver = None) -> Tuple[List[Configuration], Optional[List[Tuple[Configuration, Action]]]]:
        """
        :param observer:
        :param k:
        :return: 2-Tuple of 1. The entire RRT graph (vertices) and 2. an ordered list of state action pairs that start at the start
        and end at the goal
        """
        conf_type = type(self.G[0])
        for i in range(k):

            valid_expansion = False
            while not valid_expansion:
                n_rand = conf_type.gen_random_conf()  # This will be a valid random conf
                n_near = n_rand.nearest_vertex(self.G)
                n_next = n_rand.new_conf_from(n_near)
                valid_expansion = n_next is not None

            observer.rrt_expanded(self.G, n_rand, n_near, n_next) if observer is not None else ""

            self.G.append(n_next)

            if n_next.is_terminal():
                # We reached the goal, roll back a path
                path = [(n_next, None)]
                node = n_next
                while node.get_parent_vector() is not None:
                    path.append(node.get_parent_vector())
                    node = node.get_parent_vector()[0]

                observer.rrt_terminated(True)
                return self.G, list(path[::-1])

        observer.rrt_terminated(False)
        return self.G, None
