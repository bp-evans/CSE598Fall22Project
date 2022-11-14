import pygame
import random
import numpy as np

goalNew = (0,0)

def setValsRandom(goal):
    goalNew = goal

def genRandomObs():
    window_width, window_height = 500, 800
    obstacle_count = random.randint(6, 15)

    goal = goalNew
    goal = np.array(list((goal)))

    # rect (left, top, width, height)
    obstacle_list = list()
    while len(obstacle_list) < obstacle_count:
        top = random.randint(1, 49) * 10
        left = random.randint(1, 79) * 10
        height = random.randint(8, 50) * 10
        width = 30
        # Make sure obstacles are not on start or goal points or other obstacles
        skip_this = False
        rect = pygame.Rect(left-15, top-15, width+30, height+30)
        if rect.collidepoint(list(goal)):
            continue
        for l, t, w, h in obstacle_list:
            if rect.colliderect(pygame.Rect(l, t, w, h)):
                skip_this = True
                break
        if not skip_this:
            obstacle_list.append((left, top, width, height))
    return obstacle_list