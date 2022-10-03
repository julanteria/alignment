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


def drawGridOld(str2,str1):

    sysfont = pygame.font.get_default_font()
    font_size = 30
    font = pygame.font.SysFont(sysfont, font_size)


    
    bx = 1
    by = 1
    c = 0

    c_x = 0
    c_y = 0
    #blockSizex = math.floor(grid_width//len(str1))
    #blockSizey = math.floor(grid_width//len(str2))

    difx = 1
    dify = 1


    if len(str1) > len(str2):
        difx = len(str1)-len(str2) + 1

    elif len(str2) > len(str1):
        dify = len(str2)-len(str1) + 1

    else:
        difx = 1
        dify = 1


    for x in range(0,grid_height,blockSize):
        for y in range(0,grid_width,blockSize):

            if x <= grid_width-blockSize*dify and y <= grid_width-blockSize*(difx):
                rect = pygame.Rect(x+center_d, y+center_d, blockSize, blockSize)
                pygame.draw.rect(SCREEN, WHITE, rect, 1)
        

        if c < len(str1):
            font_x = (center_d-10)+(c+1)*blockSize
            font_y = (center_d-10)+(c)*blockSize
            img = font.render(str1[c], True, WHITE)

            #top str
            SCREEN.blit(img, (font_x,center_d-30))

        if c < len(str2):
            font_x = (center_d-10)+(c)*blockSize
            font_y = (center_d-10)+(c+1)*blockSize
            img2 = font.render(str2[c], True, WHITE)

            #left str
            SCREEN.blit(img2,(center_d-30, font_y))

        font_x = (center_d-10)+(c)*blockSize
        font_y = (center_d-10)+(c)*blockSize

        #numbers over font top
        img3_1 = font.render(str(c_x), True, WHITE)
        img3_2 = font.render(str(c_y), True, WHITE)
        #print(c)

        print(c)
        if c-1 <= len(str2):
            #left str
            SCREEN.blit(img3_1,(center_d-60, font_y))

        #top str
        SCREEN.blit(img3_2, (font_x,center_d-60))

        if c ==  0:

            img4 = font.render("i", True, WHITE)
            img5 = font.render("j", True, WHITE)

            SCREEN.blit(img4,(center_d-60, font_y-40))
            #top str
            SCREEN.blit(img5, (font_x-40,center_d-60))




        c += 1
        c_y += 1

        if c_x < len(str1):
            c_x += 1




def drawGrid(str2,str1):

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


    for c in range(len(str2)):
        font_y = (center_d-10)+(c)*blockSize
        if c ==  0:
            img4 = font.render("i", True, WHITE)
            SCREEN.blit(img4,(center_d-60, font_y))
        else:
            img4 = font.render(str(c), True, WHITE)
            SCREEN.blit(img4,(center_d-60, font_y))


        





#Pygame init stuff
global SCREEN
pygame.init()
SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
SCREEN.fill(BLACK)


#font values
sysfont = pygame.font.get_default_font()
font_size = 30
font = pygame.font.SysFont(sysfont, font_size)

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