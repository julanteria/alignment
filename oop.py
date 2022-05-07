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
        self.Costmatrix = []

        if aligmentType == "g":
            self.Costmatrix = self.getGlobalCostMatrix()


        #elif self.alignment == "l":
         #   self.Costmatrix = LocalMtarix()


    def costFunction(self, str1, str2, i, j, ):
        ch1 = str1[i - 1]
        ch2 = str2[j - 1]

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









    #def displayCount(self):
    #    print
    #    "Total Employee %d" % Employee.empCount











