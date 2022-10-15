import sys
import pygame

import Visualizers
from Configuration import StaticObstaclesConfiguration
import RRTAbstract
from Visualizers import draw_graph_and_path


def main():
    dimensions = (500, 800)
    obsnum = 12
    maph, mapw = dimensions

    goal = (650, 200)
    start = (50, 100)

    pygame.init()
    map = pygame.display.set_mode((mapw, maph))  # setting up pygame window of peoper dimensions

    print("starting configs")
    # Starting configuration
    start_config = StaticObstaclesConfiguration(start, goal)
    print(start_config.agent)
    core = RRTAbstract.RRT_Core([start_config])

    # Make this interactive
    class Observer(RRTAbstract.RRTObserver):
        def __init__(self, interactive: bool):
            self.i = 0
            self.interactive = interactive

        def rrt_expanded(self, vertices, rand: StaticObstaclesConfiguration, near: StaticObstaclesConfiguration,
                         nnext: StaticObstaclesConfiguration):
            self.i += 1
            if self.interactive or self.i % 10 == 0:
                print("Showing partial RRT tree")
                start_config.visualize(map)
                Visualizers.draw_graph_and_path(map, vertices, None)
                # Interact
                pygame.draw.circle(map, Visualizers.red, rand.agent, 5, 0)
                pygame.draw.circle(map, Visualizers.green, near.agent, 4, 0)
                pygame.draw.circle(map, Visualizers.purple, nnext.agent, 3, 0) if nnext is not None else """"""
                pygame.display.update()
                pygame.event.wait(1)
                waiting = self.interactive
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                            waiting = False

    observer = Observer(interactive=False)
    graph, path_to_goal = core.RRTAlg(1000, observer)

    start_config.visualize(map)

    draw_graph_and_path(map, graph, path_to_goal)

    # for g in graph:
    #     print(g.state)
    #     pygame.draw.circle(map, blue, (g.state[0], g.state[1]), 2, 0)
    #     if g != start_config:
    #         pygame.draw.line(map, blue, (g.state[0], g.state[1]), (g.parent.state[0], g.parent.state[1]),
    #                     1)
    #     pygame.display.update()
    # if(found_goal):
    #     print("Goal found!")
    #     for p in path_to_goal:
    #         pygame.draw.circle(map, red, (p.state[0], p.state[1]), 2, 0)
    #         pygame.display.update()

    pygame.display.update()
    pygame.display.flip()
    # pygame.event.clear()
    # pygame.event.wait(0)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
                pygame.quit()
                sys.exit()


main()
