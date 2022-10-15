from Configuration import StaticObstaclesConfiguration
import random
import math

class RRT_Core:
    def __init__(self, seed_tree, obstacles, start, goal, mapw, maph):
        self.G = seed_tree # Start off our RRT tree with the seed tree, I'm leaving this data structure undefined for now,
        # maybe we would like it to be a heap? or maybe a just a flat list? unsure which would be most performant right now,
        # or maybe it's just a Conf with children, but currently Conf doesn't store children so...
        self.obstacles = obstacles
        self.goal = goal
        self.start = start
        self.mapw = mapw
        self.reachedGoal = False
        self.maph = maph
        self.path_to_goal = []
        pass

    def add_node(self, n_next):
        self.G.append(n_next)

    def add_edge(self, nnext, nnear):
        nnext.parent = nnear
        return nnext

    def crossObstacle(self, x1, x2, y1, y2): # move incrementally in a straight line between two nodes (x1,y1) and (x2,y2) to see if a object blocks a potential edge between them
        obs = self.obstacles
        prev_x, prev_y = x1, y1
        for rectang in obs:
            for i in range(0, 101):
                u = i / 100
                x = x1 * u + x2 * (1 - u)
                y = y1 * u + y2 * (1 - u)
                if rectang.collidepoint(x, y):
                    return (prev_x, prev_y)
                prev_x, prev_y = x, y
        return x2, y2

    def is_valid(self, x, y):
        if (x <= 0 or y >= self.maph or y <=0 or x >= self.mapw):
            return False
        for o in self.obstacles:
            if(o.collidepoint(x,y)):
                return False
        return True

    def gen_random_conf(self, obstacles, goal):
        obs = obstacles # copy the obstacles list
        g = goal
        isValid = False
        while(not isValid):
            x = int(random.uniform(0, self.mapw))
            y = int(random.uniform(0, self.maph))
            isValid = True
            if(not self.is_valid(x,y)):
                isValid = False
            
        ret_config = StaticObstaclesConfiguration((x,y), g)
        print("Valid")
        return ret_config

    def distance(self, x1, y1, x2, y2): # calculate the euclidean distance between 2 nodes and return
        px = (float(x1) - float(x2)) ** 2
        py = (float(y1) - float(y2)) ** 2
        return (px + py) ** (0.5)

    def nearest_vertex(self, graph, nrand):
        dmin = 999
        near = nrand;
        for node in graph:
            x1 = node.state[0]
            y1 = node.state[1]
            x2 = nrand.state[0]
            y2 = nrand.state[1]
            dist = abs(self.distance(x1,y1,x2,y2))
            if (dist < dmin):
                near = node
                dmin = dist
        return near

    def new_conf_from(self, nrand, nnear, dmax):
        (px, py) = (nrand.state[0] - nnear.state[0], nrand.state[1] - nnear.state[1])
        theta = math.atan2(py, px)
        (x, y) = (int(nnear.state[0] + dmax * math.cos(theta)),
                  int(nnear.state[1] + dmax * math.sin(theta)))
        (x,y) = self.crossObstacle(nnear.state[0], x, nnear.state[1], y)
        new_config = StaticObstaclesConfiguration((x,y), self.goal)
        if(abs(self.goal[0] - x) <= 20 and abs(self.goal[1] - y) <= 20):
            self.reachedGoal = True
        return new_config

    def goal_reached(self):
        curr_node = self.G[len(self.G)-1]
        while(curr_node != self.start):
            self.path_to_goal.insert(0, curr_node)
            curr_node = curr_node.parent



    def RRTAlg(self, k: int, dmax=30):  # -> Graph/Tree
        self.G.append(self.start)
        for i in range(k):
            # TODO: Change to extract Configuration type from self.G
            # For example, type = type(self.G.anyNode)
            #conf_type = StaticObstaclesConfiguration((0.0,0.0),self.goal,0,0) # PLACEHOLDER
            print('Gen nrand')
            n_rand = self.gen_random_conf(self.obstacles, self.goal)  # This will be a valid random conf (passing obstacles to check for collisions)
            print('Gen nnear')
            n_near = self.nearest_vertex(self.G, n_rand) # Make sure the input type agrees, change Configuration if you want
            print("END RRT LOOP")
            n_next = self.new_conf_from(n_rand, n_near, dmax)
            if(self.is_valid(n_next.state[0], n_next.state[1])):
                n_next = self.add_edge(n_next, n_near)
                self.add_node(n_next)
                if(self.reachedGoal == True):
                    self.goal_reached()
                    return  self.G, self.path_to_goal, True
            #self.add_edge(n_next, n_near) # Adding edge will be dedicated to adding that parent of n_near to n_next
            

            # if (self.check_valid_state(n_rand)):
            #     n_near = static_config.nearest_vertex(n_rand)
            #     n_next = static_config.new_conf_from(n_rand, n_near, dmax)
            #     self.G.add_vertex(n_next)
            #     self.G.add_edge(n_next)
         
        return self.G, self.path_to_goal, False  # Some layer above this will visualize the final graph


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
