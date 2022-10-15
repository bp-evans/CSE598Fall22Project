from abc import ABC
import pygame
from Configuration import Configuration
from Configuration import StaticObstaclesConfiguration
import RRTAbstract
from abc import ABC, abstractmethod

def main():
    dimensions = (500,800)
    obsnum = 12
    maph,mapw = dimensions

    goal = (700, 350)
    start = (50, 50)

    pygame.init()
    map = pygame.display.set_mode((mapw, maph)) # setting up pygame window of peoper dimensions
    map.fill((255, 255, 255))

    grey = (70, 70, 70)
    blue = (0, 0, 255)
    red = (255, 0, 0)

    pygame.draw.circle(map, blue, start, 5, 0)
    pygame.draw.circle(map, red, goal, 20, 1)

    # Obstacles init
    obstacles = []

    # Static obstacles
    
    rect1 = pygame.Rect(100, 0, 30, 200)
    pygame.draw.rect(map, grey, rect1)
    obstacles.append(rect1)
    rect2 = pygame.Rect(50, 350, 30, 30)
    pygame.draw.rect(map, grey, rect2)
    obstacles.append(rect2)
    rect3 = pygame.Rect(250, 300, 30, 200)
    pygame.draw.rect(map, grey, rect3)
    obstacles.append(rect3)
    rect4 = pygame.Rect(250, 0, 30, 150)
    pygame.draw.rect(map, grey, rect4)
    obstacles.append(rect4)
    rect5 = pygame.Rect(450, 0, 30, 400)
    pygame.draw.rect(map, grey, rect5)
    obstacles.append(rect5)
    rect6 = pygame.Rect(550, 200, 30, 30)
    pygame.draw.rect(map, grey, rect6)
    obstacles.append(rect6)
    rect7 = pygame.Rect(575, 400, 30, 30)
    pygame.draw.rect(map, grey, rect7)
    obstacles.append(rect7)
    rect8 = pygame.Rect(650, 300, 30, 30)
    pygame.draw.rect(map, grey,rect8)
    obstacles.append(rect8)
    rect9 = pygame.Rect(650, 70, 30, 30)
    pygame.draw.rect(map, grey, rect9)
    obstacles.append(rect9)
    

    print("starting configs")
    # Starting configuration
    start_config = StaticObstaclesConfiguration(start, goal)
    print(start_config.state)
    core = RRTAbstract.RRT_Core([], obstacles, start_config, goal, 800, 500) # Empty start tree (BC not yet implemented)
    graph = []
    graph, path_to_goal, found_goal = core.RRTAlg(500)
    for g in graph:
        print(g.state)
        pygame.draw.circle(map, blue, (g.state[0], g.state[1]), 2, 0)
        if g != start_config:
            pygame.draw.line(map, blue, (g.state[0], g.state[1]), (g.parent.state[0], g.parent.state[1]),
                        1)
        pygame.display.update()
    if(found_goal):
        print("Goal found!")
        for p in path_to_goal:
            pygame.draw.circle(map, red, (p.state[0], p.state[1]), 2, 0)
            pygame.display.update()





    pygame.display.update()
    pygame.display.flip()
    #pygame.event.clear()
    #pygame.event.wait(0)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

main()

