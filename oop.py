import numpy as np


class alignment:

    # alignment attributes
    def __init__(self, string1, string2, ma, i, d, mi, wg, ws, aligmentType):
        self.string1 = string1
        self.string2 = string2

        self.matchCost = ma
        self.insertCost = i
        self.deleteCost = d
        self.missCost = mi
        self.wg = wg
        self.ws = ws

        self.gapOpening = wg
        self.gapExtension = ws

        self.aligmentType = aligmentType

        self.globalCostmatrix = []
        self.localCostmatrix = []
        self.semiglobalCostmatrix = []
        self.affineCostMatrixes = []

        self.globalAlignment = []
        self.semiglobalAlignment = []
        self.localAlignment = []
        self.globalAlignmentAffineCost = []

        if aligmentType == "g":
            self.globalCostmatrix = self.getGlobalCostMatrix()
            self.globalAlignment = self.getGlobalTraceback()

        if aligmentType == "s":
            self.semiglobalCostmatrix = self.getSemiglobalCostmatrix()
            self.semiglobalAlignment = self.getSemiGlobalTraceback()

        if aligmentType == "l":
            self.localCostmatrix = self.getLocalCostMatrix()
            self.localAlignment = self.getLocalTraceback()

        if aligmentType == "af":
            self.affineCostMatrixes = self.getAffineCostMatrix()
            self.affineCostMatrix_D = self.affineCostMatrixes[3]
            self.globalAlignmentAffineCost = self.getGlobalAffineTraceback()

    # alignment score function for minimazation optimization
    def costFunction(self, str1, str2, i, j, ):

        ch1 = str1[i - 1]
        ch2 = str2[j - 1]

        if ch1 == ch2:
            return self.matchCost

        if ch1 != ch2:
            return self.missCost

    def costFunctionX(self, str1, str2, i, j, ):
        print("i: " + str(i))
        print("j: " + str(j))
        print("D[i][j]" + str(self.affineCostMatrix_D[i][j]))
        print("D[i-1][j-1]" + str(self.affineCostMatrix_D[i-1][j-1]))
        ch1 = str1[i - 1]
        ch2 = str2[j - 1]

        if ch1 == ch2:
            print("match" + str(self.matchCost))
            print()
            return self.matchCost

        if ch1 != ch2:
            print("miss" + str(self.missCost))
            print()
            return self.missCost

    # alignment score function for maximization optimization
    def ScoreFunction(self, str1, str2, i, j):

        ch1 = str1[i - 1]
        ch2 = str2[j - 1]

        if ch1 == ch2:
            return self.matchCost

        if ch1 != ch2:
            return self.missCost


    @staticmethod
    def costumMin(list):
        x = []
        for i in list:
            if i >= 0:
                x.append(i)

        return min(x)


    # Returns two global alignment strings from global cost matrix
    def getGlobalTraceback(self):
        s1_out = ""
        s2_out = ""

        i = len(self.string1)
        j = len(self.string2)

        # build up output strings as long as loop indices i and j are bigger 0
        while i > 0 and j > 0:

            # entry D[i][j] resulted from D[i-1][j-1] + costFunction(string1,string2, i,j) => write chars under each other
            if self.globalCostmatrix[i][j] == self.globalCostmatrix[i - 1][j - 1] + self.costFunction(self.string1,
                                                                                                      self.string2, i,
                                                                                                      j):
                s1_out = self.string1[i - 1] + s1_out
                s2_out = self.string2[j - 1] + s2_out
                i -= 1
                j -= 1

            # entry D[i][j] resulted from D[i][j-1] + insertCost => add "-" (gap-character) to s1_out and add char from string2 to s2_out
            elif self.globalCostmatrix[i][j] == self.globalCostmatrix[i][j - 1] + self.insertCost:
                s1_out = "-" + s1_out
                s2_out = self.string2[j - 1] + s2_out
                j -= 1

            # entry D[i][j] resulted from D[i][j-1] + deleteCost => add "-" (gap-character) to s2_out and add char from string1 to s1_out
            else:
                s1_out = self.string1[i - 1] + s1_out
                s2_out = "-" + s2_out
                i -= 1

        # fill up s1_out with "-" (gap-character), so alignment strings are equal length
        while i > 0:
            s1_out = self.string1[i - 1] + s1_out
            s2_out = "-" + s2_out
            i -= 1

        # fill up s2_out with "-" (gap-character), so alignment strings are equal length
        while j > 0:
            s1_out = "-" + s1_out
            s2_out = self.string2[j - 1] + s2_out
            j -= 1

        return [s1_out, s2_out]

    # Returns two semi global alignment strings from semi global cost matrix
    def getSemiGlobalTraceback(self):
        s1_out = ""
        s2_out = ""

        i = len(self.string1)
        j = len(self.string2)

        # build up output strings as long as loop indices i and j are bigger 0
        while i > 0 and j > 0:
            # entry D[i][j] resulted from D[i-1][j-1] + costFunction(string1,string2, i,j) => write chars under each other
            if self.semiglobalCostmatrix[i][j] == self.semiglobalCostmatrix[i - 1][j - 1] + self.costFunction(
                    self.string1, self.string2, i, j):
                s1_out = self.string1[i - 1] + s1_out
                s2_out = self.string2[j - 1] + s2_out
                i -= 1
                j -= 1

            # entry D[i][j] resulted from D[i][j-1] + insertCost => add "-" (gap-character) to s1_out and add char from string2 to s2_out
            elif self.semiglobalCostmatrix[i][j] == self.semiglobalCostmatrix[i][j - 1] + self.insertCost:
                s1_out = "-" + s1_out
                s2_out = self.string2[j - 1] + s2_out
                j -= 1

            # entry D[i][j] resulted from D[i][j-1] + deleteCost => add "-" (gap-character) to s2_out and add char from string1 to s1_out
            else:
                s1_out = self.string1[i - 1] + s1_out
                s2_out = "-" + s2_out
                i -= 1

        # fill up s1_out with "-" (gap-character), so alignment strings are equal length
        while i > 0:
            s1_out = self.string1[i - 1] + s1_out
            s2_out = "-" + s2_out
            i -= 1

        # fill up s2_out with "-" (gap-character), so alignment strings are equal length
        while j > 0:
            s1_out = "-" + s1_out
            s2_out = self.string2[j - 1] + s2_out
            j -= 1

        return [s1_out, s2_out]

    # Returns two local alignment strings from lcoal cost matrix
    def getLocalTraceback(self):

        s1_out = ""
        s2_out = ""
        out = []

        # find indexes of max values in local cost matrix
        indexesMax = np.where(self.localCostmatrix == np.amax(self.localCostmatrix))

        # i cordinates for max values
        i_cors_max = list(indexesMax[0])
        # j cordinates for max values
        j_cors_max = list(indexesMax[1])

        # loop through all max values, each max value results in a different local alignemnt
        for ii in range(len(i_cors_max)):

            i = int(i_cors_max[ii])
            j = int(j_cors_max[ii])

            # variable to check for zero entry
            t = self.localCostmatrix[i][j]

            # build up output strings until we find a zero
            while t != 0:
                # entry D[i][j] resulted from D[i-1][j-1] + costFunction(string1,string2, i,j) => write chars under each other
                if self.localCostmatrix[i][j] == self.localCostmatrix[i - 1][j - 1] + self.ScoreFunction(self.string1,
                                                                                                         self.string2,
                                                                                                         i, j):
                    s1_out = self.string1[i - 1] + s1_out
                    s2_out = self.string2[j - 1] + s2_out
                    t = self.localCostmatrix[i - 1][j - 1]
                    i -= 1
                    j -= 1

                # entry D[i][j] resulted from D[i][j-1] + insertCost => add "-" (gap-character) to s1_out and add char from string2 to s2_out
                elif self.localCostmatrix[i][j] == self.localCostmatrix[i][j - 1] + self.insertCost:
                    s1_out = "-" + s1_out
                    s2_out = self.string2[j - 1] + s2_out
                    t = self.localCostmatrix[i][j - 1]
                    j -= 1

                # entry D[i][j] resulted from D[i][j-1] + deleteCost => add "-" (gap-character) to s2_out and add char from string1 to s1_out
                else:
                    s1_out = self.string1[i - 1] + s1_out
                    s2_out = "-" + s2_out
                    t = self.localCostmatrix[i - 1][j]
                    i -= 1

            out.append(s1_out)
            out.append(s2_out)
            s1_out = ""
            s2_out = ""

        return out


    # Returns two global alignment strings from global affine Cost Matrixes
    def getGlobalAffineTraceback(self):
        s1_out = ""
        s2_out = ""

        i = len(self.string1)
        j = len(self.string2)

        # build up output strings as long as loop indices i and j are bigger 0
        while i > 0 and j > 0:

            # entry D[i][j] resulted from G
            if self.affineCostMatrix_D[i][j] == self.affineCostMatrixes[2][i-1][j-1]:
                s1_out = self.string1[i - 1] + s1_out
                s2_out = self.string2[j - 1] + s2_out
                i -= 1
                j -= 1

            # entry D[i][j] resulted from F
            elif self.affineCostMatrix_D[i][j] == self.affineCostMatrixes[1][i][j-1]:
                s1_out = "-" + s1_out
                s2_out = self.string2[j - 1] + s2_out
                j -= 1

            # entry D[i][j] resulted from E
            else:
                s1_out = self.string1[i - 1] + s1_out
                s2_out = "-" + s2_out
                i -= 1

        # fill up s1_out with "-" (gap-character), so alignment strings are equal length
        while i > 0:
            s1_out = self.string1[i - 1] + s1_out
            s2_out = "-" + s2_out
            i -= 1

        # fill up s2_out with "-" (gap-character), so alignment strings are equal length
        while j > 0:
            s1_out = "-" + s1_out
            s2_out = self.string2[j - 1] + s2_out
            j -= 1

        return [s1_out, s2_out]

    # Returns numpy global Costmatrix
    def getGlobalCostMatrix(self):
        l1 = len(self.string1) + 1
        l2 = len(self.string2) + 1
        # initializes Numpy Matrix with zeros
        D = np.zeros(shape=(l1, l2)).astype('int')

        # adds insertCost to every entry in first row
        for i in range(1, l2):
            X = D[0][i - 1] + self.insertCost
            D[0][i] = X

        # adds insertCost to every entry in first row
        for j in range(1, l1):
            X = D[j - 1][0] + self.deleteCost
            D[j][0] = X

        # loops through all matrix entrys except D[0][0]
        # fills all entrtys according to alignment rules and custom Insertion-, Deletion- and Missmatch-Cost
        # calls function CostFunction(str1,str2,i,j)
        for i in range(1, l1):
            for j in range(1, l2):
                right = D[i][j - 1] + self.deleteCost

                down = D[i - 1][j] + self.insertCost

                diag = D[i - 1][j - 1] + self.costFunction(self.string1, self.string2, i, j)

                D[i][j] = min(right, down, diag)

        return D

    # Returns numpy semi global Costmatrix
    def getSemiglobalCostmatrix(self):
        l1 = len(self.string1) + 1
        l2 = len(self.string2) + 1

        # initializes Numpy Matrix with zeros
        D = np.zeros(shape=(l1, l2)).astype('int')

        # initializes first row
        for j in range(1, l2):
            X = D[0][j - 1] + self.insertCost
            D[0][j] = X

        # initializes first column
        # only zeros because of semi-global
        for i in range(1, l1):
            D[i][0] = 0

        # loops through all matrix entrys except D[0][0]
        # fills all entrtys according to alignment rules and custom Insertion-, Deletion- and Missmatch-Cost
        # calls function CostFunction(str1,str2,i,j)
        out = ""
        for i in range(1, l1):

            for j in range(1, l2):

                # zero cost if we are at the end of str2 because of semi-global
                if j == len(self.string2):
                    down = D[i - 1][j] + 0

                else:
                    down = D[i - 1][j] + self.insertCost

                right = D[i][j - 1] + self.deleteCost

                diag = D[i - 1][j - 1] + self.costFunction(self.string1, self.string2, i, j)

                D[i][j] = min(right, down, diag)

        return D

    # https://de.wikipedia.org/wiki/Smith-Waterman-Algorithmus#:~:text=Der%20Smith%2DWaterman%2DAlgorithmus%20ist,Alignment%20zwischen%20zwei%20Sequenzen%20berechnet.
    # python doAlign.py GGTTGACTA TGTTACGG  3 -2 -2 -3 l
    # Returns numpy local Costmatrix
    def getLocalCostMatrix(self):
        l1 = len(self.string1) + 1
        l2 = len(self.string2) + 1
        # initializes Numpy Matrix with zeros
        L = np.zeros(shape=(l1, l2)).astype('int')

        # initializes first row with zeros because of local
        for i in range(1, l2):
            L[0][i] = 0

        # initializes first column with zeros because of local
        for j in range(1, l1):
            L[j][0] = 0

        # loops through all matrix entrys except D[0][0]
        # fills all entrtys according to alignment rules and custom Insertion-, Deletion- and Missmatch-Cost
        # calls function ScoreFunction(str1,str2,i,j)
        for i in range(1, l1):
            for j in range(1, l2):

                right = L[i][j - 1] + self.deleteCost

                # no entrys smaller 0
                if right < 0:
                    right = 0

                down = L[i - 1][j] + self.insertCost

                # no entrys smaller 0
                if down < 0:
                    down = 0

                diag = L[i - 1][j - 1] + self.ScoreFunction(self.string1, self.string2, i, j)

                # no entrys smaller 0
                if diag < 0:
                    diag = 0

                L[i][j] = max(right, down, diag)

        return L

    def getAffineCostMatrix(self):
        l1 = len(self.string1) + 1
        l2 = len(self.string2) + 1

        Matrixes = []

        # initializes Numpy Matrixs with zeros
        E = np.zeros(shape=(l1, l2)).astype('int')
        F = np.zeros(shape=(l1, l2)).astype('int')
        G = np.zeros(shape=(l1, l2)).astype('int')
        D = np.zeros(shape=(l1, l2)).astype('int')

        # initializes first row
        for j in range(1, l2):
            # X = D[0][j - 1] + self.insertCost
            e = self.gapOpening + j * self.gapExtension
            f = -1  # np.nan  #undefined? numpy.nan
            g = -1  # np.nan #undefined? numpy.nan

            E[0][j] = e
            F[0][j] = f
            G[0][j] = g
            D[0][j] = self.costumMin([e,f,g])

        # initializes first column
        # only zeros because of semi-global
        for i in range(1, l1):
            e = -1
            f = self.gapOpening + i * self.gapExtension
            g = -1

            E[i][0] = e
            F[i][0] = f
            G[i][0] = g
            D[i][0] = self.costumMin([e,f,g])

        for i in range(1, l1):
            for j in range(1, l2):

                ee = E[i][j - 1]
                ef = F[i][j - 1]
                eg = G[i][j - 1]



                e_entry = self.costumMin([ee, ef, eg])

                if e_entry == ee:
                    E_e = e_entry + self.ws
                else:
                    E_e = e_entry + self.wg + self.ws

                E[i][j] = E_e

                fe = E[i - 1][j]
                ff = F[i - 1][j]
                fg = G[i - 1][j]


                f_entry = self.costumMin([fe, ff, fg])

                if f_entry == ff:
                    F_e = f_entry + self.ws
                else:
                    F_e = f_entry + self.wg + self.ws

                F[i][j] = F_e

                ge = E[i - 1][j - 1]
                gf = F[i - 1][j - 1]
                gg = G[i - 1][j - 1]

                if ge < 0:
                    ge = np.inf

                if gf < 0:
                    gf = np.inf

                if gg < 0:
                    gg = np.inf

                cost = self.costFunction(self.string1,self.string2,i,j)

                G_e = min(ge+cost,gf+cost,gg+cost)

                G[i][j] = G_e


                D[i][j] = min(G_e, E_e, F_e)

        Matrixes.append(E)
        Matrixes.append(F)
        Matrixes.append(G)
        Matrixes.append(D)

        return Matrixes














