# import pygame
#
# dimensions = (500, 800)
# obsnum = 12
# maph, mapw = dimensions
#
#
# def setup():
#     map = pygame.display.set_mode((mapw, maph))  # setting up pygame window of proper dimensions
#     map.fill((255, 255, 255))
#
#     grey = (70, 70, 70)
#
#     # Static obstacles
#     pygame.draw.rect(map, grey, (100, 0, 30, 200))
#     pygame.draw.rect(map, grey, (50, 350, 30, 30))
#     pygame.draw.rect(map, grey, (250, 300, 30, 200))
#     pygame.draw.rect(map, grey, (250, 0, 30, 150))
#     pygame.draw.rect(map, grey, (450, 0, 30, 400))
#     pygame.draw.rect(map, grey, (550, 200, 30, 30))
#     pygame.draw.rect(map, grey, (575, 400, 30, 30))
#     pygame.draw.rect(map, grey, (650, 300, 30, 30))
#     pygame.draw.rect(map, grey, (650, 70, 30, 30))
#
#     pygame.display.update()
#     # pygame.event.clear()
#     # pygame.event.wait(0)
