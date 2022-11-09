import sys
import pygame

import Visualizers
from Configuration import StaticObstaclesConfiguration
import RRTAbstract
from Visualizers import draw_graph_and_path


def main():
    """
    This is to run the RRT algorithm for testing purposes. Note that this file will not be used when running the RRT algo
    for a game Agent.
    :return:
    """
    dimensions = (500, 800)
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
                pygame.draw.circle(map, Visualizers.Color.red.value(), rand.agent, 5, 0)
                pygame.draw.circle(map, Visualizers.Color.green.value(), near.agent, 4, 0)
                pygame.draw.circle(map, Visualizers.Color.purple.value(), nnext.agent, 3, 0) if nnext is not None else """"""
                pygame.display.update()
                pygame.event.wait(1)
                waiting = self.interactive
                while waiting:
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                            waiting = False

        def rrt_terminated(self, found_terminal: bool):
            print("RRT algo terminated after " + str(self.i) + " expansions. Found terminal: " + str(found_terminal))

    observer = Observer(interactive=False)
    graph, path_to_goal = core.RRTAlg(2000, observer)

    start_config.visualize(map)

    draw_graph_and_path(map, graph, path_to_goal)

    pygame.display.update()
    pygame.display.flip()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
                pygame.quit()
                sys.exit()


main()
