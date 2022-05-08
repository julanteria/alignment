
import pygame
from pygame.locals import *
import time
import math
import sys
import numpy as np


BLACK = (0, 0, 0)
WHITE = (200, 200, 200)
RED = (255,0,0)
WINDOW_HEIGHT = 700
WINDOW_WIDTH = 1200



#Strings
global str1 
global str2

string1 = sys.argv[1]
string2 = sys.argv[2]
insCost = int(sys.argv[3])
delCost = int(sys.argv[4])
misCost = int(sys.argv[5])


str1 = string1
str2 = string2



if len(str1) > len(str2):
    n = len(str1)
else:
    n = len(str2)


grid_height = 500
grid_width = 500
center_d = (700 - grid_width)//2 #achtung anpassen evtl 700 

blockSize = math.floor(grid_width//n)



def draw_path(alig_str1, alig_str2):

    bx = 0
    by = 0
    for i in range(len(alig_str1)):
        pygame.display.update()
        time.sleep(0.3)


        if alig_str1[i] == "-":
            bx+=1
            right_arrow(bx,by)

        if alig_str2[i] == "-":
            by+=1
            down_arrow(bx,by)
            

        else:
            bx+=1
            by+=1
            diag_arrow(bx,by)

#hier stimmt was nicht
def draw_perf_path(path):
    print(path)
    bx = 0
    by = 0
    for do in path:
        print(1)
        pygame.display.update()
        time.sleep(0.1)
        if do == "r":
            bx+=1
            right_arrow(bx,by)

        if do == "d":
            by+=1
            down_arrow(bx,by)
            

        elif do == "D":
            bx+=1
            by+=1
            diag_arrow(bx,by)



def draw_alig_header(str1,str2):

    sysfont = pygame.font.get_default_font()
    font_size = 30
    font = pygame.font.SysFont(sysfont, font_size)

    for c in range(len(str2)):
        font_x = blockSize*2+center_d+blockSize*len(str2)+c*blockSize//2
        
        
        if str2[c] == "-":
            cchar = " - "
        else:
            cchar = str2[c]

        img = font.render(cchar, True, WHITE)

        #top str
        SCREEN.blit(img, (font_x,center_d-30))

    for c in range(len(str1)):
        font_x = blockSize*2+center_d+blockSize*len(str1)+c*blockSize//2
        
        if str1[c] == "-":
            cchar = " - "
        else:
            cchar = str1[c]
        img = font.render(cchar, True, WHITE)

        #top str
        SCREEN.blit(img, (font_x,center_d+30))





def drawGrid_arrows(x,y):
    down_arrows(x,y)
    right_arrows(x,y)
    diag_arrows(x,y)

def diag_arrows(x,y):

    for by in range(1,y):
        for bx in range(1,x):
            pygame.draw.polygon(surface=SCREEN, color=WHITE, points=[(center_d+8+blockSize*bx-blockSize*0.2,center_d-12+blockSize*by),(center_d+blockSize*bx,center_d+blockSize*by),(center_d-12+blockSize*bx,center_d+8+blockSize*by-blockSize*0.2)])
            pygame.draw.line(surface=SCREEN, color=WHITE, start_pos=(center_d+blockSize*(bx-1),center_d+blockSize*(by-1)), end_pos=(center_d+blockSize*(bx),center_d+blockSize*(by)))


def down_arrows(x,y):
    for by in range(1,y):
        for bx in range(0,x):
            pygame.draw.polygon(surface=SCREEN, color=WHITE, points=[(center_d+6-center_d*0.1+blockSize*bx,center_d-10-center_d*0.1+blockSize*by),(center_d-6+center_d*0.1+blockSize*bx,center_d-10-center_d*0.1+blockSize*by),(center_d+blockSize*bx,center_d+blockSize*by)])

def right_arrows(x,y):

    for by in range(0,y):
        for bx in range(1,x):
            #first point top
            #second middle
            #third down
            pygame.draw.polygon(surface=SCREEN, color=WHITE, points=[(center_d-2+blockSize*bx-blockSize*0.2,center_d-center_d*0.1+6+blockSize*by),(center_d+blockSize*bx,center_d+blockSize*by),(center_d-2+blockSize*bx-blockSize*0.2,center_d-5+center_d*0.1+blockSize*by)])


def diag_arrow(bx,by):
    pygame.draw.polygon(surface=SCREEN, color=RED, points=[(center_d+8+blockSize*bx-blockSize*0.2,center_d-12+blockSize*by),(center_d+blockSize*bx,center_d+blockSize*by),(center_d-12+blockSize*bx,center_d+8+blockSize*by-blockSize*0.2)])
    pygame.draw.line(surface=SCREEN, color=RED, start_pos=(center_d+blockSize*(bx-1),center_d+blockSize*(by-1)), end_pos=(center_d+blockSize*(bx),center_d+blockSize*(by)))



def right_arrow(bx,by):

    pygame.draw.polygon(surface=SCREEN, color=RED, points=[(center_d-2+blockSize*bx-blockSize*0.2,center_d-center_d*0.1+6+blockSize*by),(center_d+blockSize*bx,center_d+blockSize*by),(center_d-2+blockSize*bx-blockSize*0.2,center_d-5+center_d*0.1+blockSize*by)])
    pygame.draw.line(surface=SCREEN, color=RED, start_pos=(center_d+blockSize*(bx),center_d+blockSize*(by)), end_pos=(center_d+blockSize*(bx-1),center_d+blockSize*(by)))

def down_arrow(bx,by):
    pygame.draw.polygon(surface=SCREEN, color=RED, points=[(center_d+6-center_d*0.1+blockSize*bx,center_d-10-center_d*0.1+blockSize*by),(center_d-6+center_d*0.1+blockSize*bx,center_d-10-center_d*0.1+blockSize*by),(center_d+blockSize*bx,center_d+blockSize*by)])
    for i in range(300):

        pygame.draw.line(surface=SCREEN, color=RED, start_pos=(center_d+blockSize*(bx),center_d+blockSize*(by-1)), end_pos=(center_d+blockSize*(bx),center_d+blockSize*(by)))



def drawGrid(str2,str1):



    sysfont = pygame.font.get_default_font()
    t0 = time.time()
    font_size = 30
    font = pygame.font.SysFont(sysfont, font_size)
    #print(n)
    #str1 = str1.replace("-","")
    #str2 = str2.replace("-","")


    
    bx = 1
    by = 1
    c = 0
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
        img3 = font.render(str(c), True, WHITE)
        #left str
        SCREEN.blit(img3,(center_d-60, font_y))
        #top str
        SCREEN.blit(img3, (font_x,center_d-60))

        if c ==  0:

            img4 = font.render("i", True, WHITE)
            img5 = font.render("j", True, WHITE)

            SCREEN.blit(img4,(center_d-60, font_y-40))
            #top str
            SCREEN.blit(img5, (font_x-40,center_d-60))




        c += 1








def CostFunction(str1,str2,i,j):

    ch1 = str1[i-1]
    ch2 = str2[j-1]

    if ch1 == ch2:
        return 0


    if ch1 != ch2:
        return misCost



def getCostMatrix(s1,s2):
    l1 = len(s1)+1
    l2 = len(s2)+1
    D = np.zeros(shape=(l1,l2)).astype('int')

    for i in range(1,l2):
        X = D[0][i-1] + insCost
        D[0][i] = X 

    for j in range(1,l1):
        X = D[j-1][0] + delCost
        D[j][0] = X


    for i in range(1,l1):
        for j in range(1,l2):

            right = D[i][j-1] + delCost

            down = D[i-1][j] + insCost

            diag = D[i-1][j-1] + CostFunction(s1,s2,i,j)


            D[i][j] = min(right,down,diag)

    #print("Cost-Matrix for: ")
    #print("String 1: " + str(string1))
    #print("String 2: " + str(string2))
    #print()
    #print(D)
    #print()
    #print()
    #return D[l1-1][l2-1]
    return D


def getCostMatrixSemiGlobal(s1,s2):
    l1 = len(s1)+1
    l2 = len(s2)+1

    #initializes Numpy Matrix with zeros
    D = np.zeros(shape=(l1,l2)).astype('int')


    #initializes first row
    for j in range(1,l2):
        X = D[0][j-1] + insCost
        D[0][j] = X


    #initializes first column
    #only zeros because of semi-global
    for i in range(1,l1):
        D[i][0] = 0


    #loops through all matrix entrys except D[0][0]
    #fills all entrtys according to alignment rules and custom Insertion-, Deletion- and Missmatch-Cost
    #calls function CostFunction(str1,str2,i,j)
    out = ""
    for i in range(1,l1):

        for j in range(1,l2):


            #zero cost if we are at the end of str2 because of semi-global
            if j == len(s2): 
                down = D[i-1][j] + 0

            else:
                down = D[i-1][j] + insCost


            right = D[i][j-1] + delCost


            diag = D[i-1][j-1] + CostFunction(s1,s2,i,j)

            D[i][j] = min(right,down,diag)

    return D


def getAlignemt(D):
    perfpath = ""
    s1_out = ""
    s2_out = ""

    i = len(string1)
    j = len(string2)

    while i > 0 and j > 0:
        if D[i][j] == D[i-1][j-1] + CostFunction(string1,string2,i,j):
            s1_out = string1[i-1] + s1_out
            s2_out = string2[j-1] + s2_out
            i -= 1
            j -= 1

            perfpath += "D"

        elif D[i][j] == D[i][j-1] + insCost: 
            s1_out = "-" + s1_out
            s2_out = string2[j-1] + s2_out
            j -= 1
            perfpath += "r"

        else:
            s1_out = string1[i-1] + s1_out
            s2_out = "-" + s2_out
            i -= 1
            perfpath += "d"

    while i > 0:
        s1_out = string1[i-1] + s1_out
        s2_out = "-" + s2_out
        i -= 1
        perfpath += "d"


    while j > 0:
        s1_out = "-" + s1_out
        s2_out = string2[j-1] + s2_out
        j -= 1
        perfpath += "r"

    perfpath = "".join(reversed(perfpath))
    print(perfpath)
    
    return s1_out, s2_out, perfpath







def getNumberOfAlign(s1, s2):

    l1=len(s1)
    l2=len(s2)


    A = np.full((l1+1,l2+1),1)

    for i in range(1,l1+1):
        for j in range(1,l2+1):
            A[j][i] = A[j-1][i-1] + A[j][i-1] + A[j-1][i]


    #print(A)
    return A[l2][l1]



def getAlignemtStrings(str1,str2):

    D = getCostMatrixSemiGlobal(str1,str2)

    align = getAlignemt(D)

    return align




















def main():
    global SCREEN
    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    SCREEN.fill(BLACK)


    drawGrid(str1,str2)
    #drawGrid_arrows(len(str2)+1,len(str1)+1)
    pygame.display.update()

    aliStrList = getAlignemtStrings(str1,str2)

    draw_alig_header(aliStrList[1], aliStrList[0])
    draw_perf_path(aliStrList[2])
    #draw_perf_path(" ")

    while True:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.update()


if __name__ == '__main__':
    main()
















