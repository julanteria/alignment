import pygame
from pygame.locals import *
import time
import math
import sys
import numpy as np


#constant values
BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
RED = (255,0,0)
WINDOW_HEIGHT = 700
WINDOW_WIDTH = 1200

grid_height = 500
grid_width = 500
center_d = (700 - grid_width)//2 #achtung anpassen evtl 700 

str1 = sys.argv[1]
str2 = sys.argv[2]



#define the size of the blocks in the grid
if len(str1) > len(str2):
    n = len(str1)
else:
    n = len(str2)

blockSize = math.floor(grid_width//n)



def drawGrid(str2,str1):

    #What are those values?
    bx = 1
    by = 1
    c = 0

    c_x = 0
    c_y = 0

    difx = 1
    dify = 1


    if len(str1) > len(str2):
        difx = len(str1)-len(str2) + 1

    elif len(str2) > len(str1):
        dify = len(str2)-len(str1) + 1

    else:
        difx = 1
        dify = 1

    
    #draw the grid
    for x in range(0,grid_height,blockSize):
        for y in range(0,grid_width,blockSize):

            if x <= grid_width-blockSize*dify and y <= grid_width-blockSize*(difx):
                rect = pygame.Rect(x+center_d, y+center_d, blockSize, blockSize)
                pygame.draw.rect(SCREEN, WHITE, rect, 1)


def drawStrIndexes(str1,str2):
    #write the lenght of str1 as indices on the grid
    for c1 in range(len(str1)+1):
        font_y = (center_d-10)+(c1)*blockSize
        if c1 ==  0:
            img4 = font.render("i", True, WHITE)
            SCREEN.blit(img4,(center_d-60, font_y))
        else:
            img4 = font.render(str(c1), True, WHITE)
            SCREEN.blit(img4,(center_d-60, font_y))

    #write the lenght of str2 as indices on the grid
    for c2 in range(len(str2)+1):
        font_x = (center_d-10)+(c2+1)*blockSize

        if c2 ==  0:
            img4 = font.render("j", True, WHITE)
            SCREEN.blit(img4,(font_x-40,center_d-60))
        else:
            img4 = font.render(str(c2), True, WHITE)
            SCREEN.blit(img4,(font_x-40,center_d-60))



        





#Pygame init stuff
global SCREEN
pygame.init()
SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
SCREEN.fill(BLACK)


#font values
sysfont = pygame.font.get_default_font()
font_size = 30
font = pygame.font.SysFont(sysfont, font_size)

#main()

drawGrid(str1,str2)
pygame.display.update()

drawStrIndexes(str1,str2)
pygame.display.update()

while True:
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()