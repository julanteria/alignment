import numpy as np
import sys


class alignment:

    def __init__(self, string1, string2, ma, i, d, mi, aligmentType):
        self.string1 = string1
        self.string2 = string2
        self.matchCost = ma
        self.insertCost = i
        self.deleteCost = d
        self.missCost = mi
        self.aligmentType = aligmentType
        self.globalCostmatrix = []
        self.localCostmatrix = []
        self.semiglobalCostmatrix = []

        if aligmentType == "g":
            self.globalCostmatrix = self.getGlobalCostMatrix()

        if aligmentType == "s":
            self.semiglobalCostmatrix = self.getSemiglobalCostmatrix()

        if aligmentType == "l":
            self.localCostmatrix = self.getLocalCostMatrix()

        print(self.matchCost)
        print(self.insertCost)
        print(self.deleteCost)
        print(self.missCost)


        #elif self.alignment == "l":
         #   self.Costmatrix = LocalMtarix()


    def costFunction(self, str1, str2, i, j, ):
        ch1 = str1[i - 1]
        ch2 = str2[j - 1]

        if ch1 == ch2:
            return self.matchCost

        if ch1 != ch2:
            return self.missCost


    def ScoreFunction(self,str1,str2,i,j):

        ch1 = str1[i-1]
        ch2 = str2[j-1]

        if ch1 == ch2:
            return self.matchCost


        if ch1 != ch2:
            return self.missCost





    def getGlobalCostMatrix(self):
        l1 = len(self.string1) + 1
        l2 = len(self.string2) + 1
        D = np.zeros(shape=(l1, l2)).astype('int')

        for i in range(1, l2):
            X = D[0][i - 1] + self.insertCost
            D[0][i] = X

        for j in range(1, l1):
            X = D[j - 1][0] + self.deleteCost
            D[j][0] = X

        for i in range(1, l1):
            for j in range(1, l2):
                right = D[i][j - 1] + self.deleteCost

                down = D[i - 1][j] + self.insertCost

                diag = D[i - 1][j - 1] + self.costFunction(self.string1, self.string2, i, j)

                D[i][j] = min(right, down, diag)


        return D

    def getSemiglobalCostmatrix(self):
        l1 = len(self.string1)+1
        l2 = len(self.string2)+1

        #initializes Numpy Matrix with zeros
        D = np.zeros(shape=(l1,l2)).astype('int')


        #initializes first row
        for j in range(1,l2):
            X = D[0][j-1] + self.insertCost
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
                if j == len(self.string2):
                    down = D[i-1][j] + 0

                else:
                    down = D[i-1][j] + self.insertCost


                right = D[i][j-1] + self.deleteCost


                diag = D[i-1][j-1] + self.costFunction(self.string1,self.string2,i,j)

                D[i][j] = min(right,down,diag)

        return D

    # Matrix scheint zu stimmen laut https://de.wikipedia.org/wiki/Smith-Waterman-Algorithmus#:~:text=Der%20Smith%2DWaterman%2DAlgorithmus%20ist,Alignment%20zwischen%20zwei%20Sequenzen%20berechnet.
    # Backtracking ist noch falsch
    # python doAlign.py GGTTGACTA TGTTACGG  3 -2 -2 -3 l
    def getLocalCostMatrix(self):
        l1 = len(self.string1)+1
        l2 = len(self.string2)+1
        D = np.zeros(shape=(l1,l2)).astype('int')

        for i in range(1,l2):
            D[0][i] = 0 

        for j in range(1,l1):
            D[j][0] = 0


        for i in range(1,l1):
            for j in range(1,l2):

                right = D[i][j-1] + self.deleteCost

                if right < 0:
                    right = 0

                down = D[i-1][j] + self.insertCost

                if down < 0:
                    down = 0

                diag = D[i-1][j-1] + self.ScoreFunction(self.string1,self.string2,i,j)

                if diag < 0:
                    diag = 0




                D[i][j] = max(right,down,diag)

        return D














    #def displayCount(self):
    #    print
    #    "Total Employee %d" % Employee.empCount











