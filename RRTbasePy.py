# import random
# import math
# import pygame
#
#
# class RRTMap: # The game space or 2d map that the obstacles and agent are in
#     def __init__(self, start, goal, MapDimensions, obsdim, obsnum):
#         self.start = start # starting x,y
#         self.goal = goal # goal x,y
#         self.MapDimensions = MapDimensions # dimensions of the gamespace window
#         self.Maph, self.Mapw = self.MapDimensions # map width and map height
#
#         # window settings
#         self.MapWindowName = 'RRT path planning'
#         pygame.display.set_caption(self.MapWindowName)
#         self.map = pygame.display.set_mode((self.Mapw, self.Maph)) # setting up pygame window of peoper dimensions
#         self.map.fill((255, 255, 255)) # background fill of map
#         self.nodeRad = 2 # radius of nodes on screen
#         self.nodeThickness = 0
#         self.edgeThickness = 1 # thickness of edge or line between nodes when displayed
#
#         self.obstacles = [] # list to hold obstacles
#         self.obsdim = obsdim # dimensions of square obstacles
#         self.obsNumber = obsnum # number of obstacles generated
#
#         # Colors (defining colors for pygame)
#         self.grey = (70, 70, 70)
#         self.Blue = (0, 0, 255)
#         self.Green = (0, 255, 0)
#         self.Red = (255, 0, 0)
#         self.white = (255, 255, 255)
#
#     def drawMap(self, obstacles): # draw the goal and start on the map as well as all obstacles
#         pygame.draw.circle(self.map, self.Green, self.start, self.nodeRad + 5, 0)
#         pygame.draw.circle(self.map, self.Green, self.goal, self.nodeRad + 20, 1)
#         self.drawObs(obstacles)
#
#     def drawPath(self, path): # highlight every node in the shortest path from start to goal red
#         for node in path:
#             pygame.draw.circle(self.map, self.Red, node, 3, 0)
#
#     def drawObs(self, obstacles): # draw the obstacles in the game space
#         obstaclesList = obstacles.copy()
#         while (len(obstaclesList) > 0):
#             obstacle = obstaclesList.pop(0)
#             pygame.draw.rect(self.map, self.grey, obstacle)
#
#
# class RRTGraph: # the RTT graph class
#     def __init__(self, start, goal, MapDimensions, obsdim, obsnum):
#         (x, y) = start
#         self.start = start
#         self.goal = goal
#         self.goalFlag = False # flag indicating if goal achieved
#         self.maph, self.mapw = MapDimensions
#         self.x = []
#         self.y = []
#         self.parent = [] # list of parents for nodes
#         # initialize the tree
#         self.x.append(x) # add start x
#         self.y.append(y) # add start y
#         self.parent.append(0) # add first node (0 in this case) to the parent list
#         # the obstacles
#         self.obstacles = []
#         self.obsDim = obsdim
#         self.obsNum = obsnum
#         # path
#         self.goalstate = None
#         self.path = [] # list to hold path
#
#     def makeRandomRect(self): # geberate random rectangle for obstacles
#         uppercornerx = int(random.uniform(0, self.mapw - self.obsDim))
#         uppercornery = int(random.uniform(0, self.maph - self.obsDim))
#
#         return (uppercornerx, uppercornery)
#
#     def makeobs(self): # create rectangles for game space to serve as obstacles, makes sure that the obstacles don't overlap with start or goal
#         obs = []
#         for i in range(0, self.obsNum):
#             rectang = None
#             startgoalcol = True
#             while startgoalcol:
#                 upper = self.makeRandomRect()
#                 rectang = pygame.Rect(upper, (self.obsDim, self.obsDim))
#                 if rectang.collidepoint(self.start) or rectang.collidepoint(self.goal):
#                     startgoalcol = True
#                 else:
#                     startgoalcol = False
#             obs.append(rectang)
#         self.obstacles = obs.copy()
#         return obs
#
#     def add_node(self, n, x, y): # add new node (n the node number, x and y the node coordinates)
#         self.x.insert(n, x)
#         self.y.insert(n, y) # this is edited
#
#     def remove_node(self, n): # remove node from graph by popping x and y from respecitice x and y lists at index of node
#         self.x.pop(n)
#         self.y.pop(n)
#
#     def add_edge(self, parent, child): # add an edge between two nodes by adding in the parent node at the index of the child node (i.e., the parent of node 5 is the node stored in the parent list at index 5)
#         self.parent.insert(child, parent)
#
#     def remove_edge(self, n): # pop the parent of the node to remove the virtual edge
#         self.parent.pop(n)
#
#     def number_of_nodes(self): # get the total number of nodes in the RRT graph
#         return len(self.x)
#
#     def distance(self, n1, n2): # calculate the euclidean distance between 2 nodes and return
#         (x1, y1) = (self.x[n1], self.y[n1])
#         (x2, y2) = (self.x[n2], self.y[n2])
#         px = (float(x1) - float(x2)) ** 2
#         py = (float(y1) - float(y2)) ** 2
#         return (px + py) ** (0.5)
#
#     def sample_envir(self): # randomly sample the environment to get a random x and y coordinate for a random node and return
#         x = int(random.uniform(0, self.mapw))
#         y = int(random.uniform(0, self.maph))
#         return x, y
#
#     def nearest(self, n): # loop through all current nodes to find the one that is the closest to the argumeent node (argument node is generally nrand)
#         dmin = self.distance(0, n)
#         nnear = 0
#         for i in range(0, n):
#             if self.distance(i, n) < dmin:
#                 dmin = self.distance(i, n)
#                 nnear = i
#         return nnear
#
#     def isFree(self): # check if any of the obstacles in the game world collide with the current x and y of the most recently added node to see if it is "free" to use
#         n = self.number_of_nodes() - 1
#         (x, y) = (self.x[n], self.y[n])
#         obs = self.obstacles.copy()
#         while len(obs) > 0:
#             rectang = obs.pop(0)
#             if rectang.collidepoint(x, y):
#                 self.remove_node(n)
#                 return False
#         return True
#
#     def crossObstacle(self, x1, x2, y1, y2): # move incrementally in a straight line between two nodes (x1,y1) and (x2,y2) to see if a object blocks a potential edge between them
#         obs = self.obstacles.copy()
#         while (len(obs) > 0):
#             rectang = obs.pop(0)
#             for i in range(0, 101):
#                 u = i / 100
#                 x = x1 * u + x2 * (1 - u)
#                 y = y1 * u + y2 * (1 - u)
#                 if rectang.collidepoint(x, y):
#                     return True
#         return False
#
#     def connect(self, n1, n2): # connect two nodes by adding a edge between them if path is not blocked by an obstacle, else remove the node we are trying to connect to from node list
#         (x1, y1) = (self.x[n1], self.y[n1])
#         (x2, y2) = (self.x[n2], self.y[n2])
#         if self.crossObstacle(x1, x2, y1, y2):
#             self.remove_node(n2)
#             return False
#         else:
#             self.add_edge(n1, n2)
#             return True
#
#     def step(self, nnear, nrand, dmax=25): # the part of RRT that takes the random node, the nearest node, and makes a dmax step in the direction of nrand and creates a new node
#         d = self.distance(nnear, nrand)
#         if d > dmax:
#             u = dmax / d
#             (xnear, ynear) = (self.x[nnear], self.y[nnear])
#             (xrand, yrand) = (self.x[nrand], self.y[nrand])
#             (px, py) = (xrand - xnear, yrand - ynear)
#             theta = math.atan2(py, px)
#             (x, y) = (int(xnear + dmax * math.cos(theta)),
#                       int(ynear + dmax * math.sin(theta)))
#             self.remove_node(nrand)
#             if abs(x - self.goal[0]) <= dmax and abs(y - self.goal[1]) <= dmax:
#                 self.add_node(nrand, self.goal[0], self.goal[1])
#                 self.goalstate = nrand
#                 self.goalFlag = True
#             else:
#                 self.add_node(nrand, x, y)
#
#     def RRT_core(self): # this is a single iteration of the base RRT algorithm, it selectes a random node, finds the nearest, finds the new node, adds the edge and adds the vertext to the tree
#         n = self.number_of_nodes()
#         x, y = self.sample_envir() # finding nrand
#         self.add_node(n, x, y) # add the vertex
#         if self.isFree():
#             xnearest = self.nearest(n) # finding nearest
#             self.step(xnearest, n) # getting node dmax step from nearest in direction of nrand
#             self.connect(xnearest, n) # add the edge
#         return self.x, self.y, self.parent # return the x, y, and parent of new node (parent is the node the current n will form and edge with)
#
#     def path_to_goal(self): # work backwards from the goal and add nodes to the path until reaching the node 0 or the start node
#         if self.goalFlag:
#             self.path = []
#             self.path.append(self.goalstate)
#             newpos = self.parent[self.goalstate]
#             while (newpos != 0):
#                 self.path.append(newpos)
#                 newpos = self.parent[newpos]
#             self.path.append(0)
#         return self.goalFlag
#
#     def getPathCoords(self): # create a list of path cords which contains at each index the x,y coords of the nodes contained in self.parent
#         pathCoords = []
#         for node in self.path:
#             x, y = (self.x[node], self.y[node])
#             pathCoords.append((x, y))
#         return pathCoords
#
#     def cost(self, n): # calculates the cumulative cost from or distance travelled along the final path from start to goal
#         ninit = 0
#         n = n
#         parent = self.parent[n]
#         c = 0
#         while n is not ninit:
#             c = c + self.distance(n, parent)
#             n = parent
#             if n is not ninit:
#                 parent = self.parent[n]
#         return c
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
