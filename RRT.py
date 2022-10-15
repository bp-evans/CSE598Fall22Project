# import pygame
# from RRTbasePy import RRTGraph
# from RRTbasePy import RRTMap
# import time
#
# def main():
#     # setting the arguments needed to create the RRTGraph and RRTMap
#     dimensions =(1000,2000)
#     start=(50,50)
#     goal=(800,300)
#     obsdim=30
#     obsnum=200
#     iteration=0
#     t1=0
#
#     pygame.init()
#     map=RRTMap(start,goal,dimensions,obsdim,obsnum)
#     graph=RRTGraph(start,goal,dimensions,obsdim,obsnum)
#
#     obstacles=graph.makeobs()
#     map.drawMap(obstacles)
#
#     t1=time.time()
#
#     """
#     This loop below calls the core RTT algorithm until the goal is achieved. The other parts of the code are concerned with timing and drawing the results in pygame
#     """
#     while (not graph.path_to_goal()): # run while the goal is not reached
#         time.sleep(0.005)
#         elapsed=time.time()-t1
#         t1=time.time()
#         #raise exception if timeout
#         if elapsed > 10:
#             print('timeout re-initiating the calculations')
#             raise
#
#         # Calling the expand which is the base RRT algorithm
#         X, Y, Parent = graph.RRT_core()
#         # Based on return from one RRT alg iteration, plot the result in pygame
#         pygame.draw.circle(map.map, map.grey, (X[-1], Y[-1]), map.nodeRad*2, 0)
#         pygame.draw.line(map.map, map.Blue, (X[-1], Y[-1]), (X[Parent[-1]], Y[Parent[-1]]),
#                         map.edgeThickness)
#
#         # Show updated RRT graph in game window every  5 iterations
#         if iteration % 5 == 0:
#             pygame.display.update()
#         iteration += 1
#     # In game, visually show the path calculated
#     print("Path found, drawing...")
#     map.drawPath(graph.getPathCoords())
#     # Required pygame stuff to run window
#     pygame.display.update()
#     pygame.event.clear()
#     pygame.event.wait(0)
#
#
#
# if __name__ == '__main__':
#     result=False
#     while not result:
#         try:
#             main()
#             result=True
#         except:
#             result=False
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
