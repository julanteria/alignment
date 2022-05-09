import numpy as np


class alignment:

    #alignment attributes
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

        self.globalAlignment = []
        self.semiglobalAlignment = []
        self.localAlignment = []

        if aligmentType == "g":
            self.globalCostmatrix = self.getGlobalCostMatrix()
            self.globalAlignment = self.getGlobalTraceback()

        if aligmentType == "s":
            self.semiglobalCostmatrix = self.getSemiglobalCostmatrix()
            self.semiglobalAlignment = self.getSemiGlobalTraceback()

        if aligmentType == "l":
            self.localCostmatrix = self.getLocalCostMatrix()
            self.localAlignment = self.getLocalTraceback()


    #alignment score function for maximization optimization
    def costFunction(self, str1, str2, i, j, ):
        ch1 = str1[i - 1]
        ch2 = str2[j - 1]

        if ch1 == ch2:
            return self.matchCost

        if ch1 != ch2:
            return self.missCost


    #alignment score function for maximization optimization
    def ScoreFunction(self, str1, str2, i, j):

        ch1 = str1[i - 1]
        ch2 = str2[j - 1]

        if ch1 == ch2:
            return self.matchCost

        if ch1 != ch2:
            return self.missCost
    

    #Returns two global alignment strings from global cost matrix
    def getGlobalTraceback(self):
        s1_out = ""
        s2_out = ""

        i = len(self.string1)
        j = len(self.string2)


        #build up output strings as long as loop indices i and j are bigger 0
        while i > 0 and j > 0:
            
            #entry D[i][j] resulted from D[i-1][j-1] + costFunction(string1,string2, i,j) => write chars under each other
            if self.globalCostmatrix[i][j] == self.globalCostmatrix[i - 1][j - 1] + self.costFunction(self.string1,self.string2, i,j):
                s1_out = self.string1[i - 1] + s1_out
                s2_out = self.string2[j - 1] + s2_out
                i -= 1
                j -= 1

            #entry D[i][j] resulted from D[i][j-1] + insertCost => add "-" (gap-character) to s1_out and add char from string2 to s2_out
            elif self.globalCostmatrix[i][j] == self.globalCostmatrix[i][j - 1] + self.insertCost:
                s1_out = "-" + s1_out
                s2_out = self.string2[j - 1] + s2_out
                j -= 1

            #entry D[i][j] resulted from D[i][j-1] + deleteCost => add "-" (gap-character) to s2_out and add char from string1 to s1_out
            else:
                s1_out = self.string1[i - 1] + s1_out
                s2_out = "-" + s2_out
                i -= 1

        #fill up s1_out with "-" (gap-character), so alignment strings are equal length
        while i > 0:
            s1_out = self.string1[i - 1] + s1_out
            s2_out = "-" + s2_out
            i -= 1

        #fill up s2_out with "-" (gap-character), so alignment strings are equal length
        while j > 0:
            s1_out = "-" + s1_out
            s2_out = self.string2[j - 1] + s2_out
            j -= 1

        return [s1_out, s2_out]

    #Returns two semi global alignment strings from semi global cost matrix
    def getSemiGlobalTraceback(self):
        s1_out = ""
        s2_out = ""

        i = len(self.string1)
        j = len(self.string2)

       #build up output strings as long as loop indices i and j are bigger 0     
        while i > 0 and j > 0:
            #entry D[i][j] resulted from D[i-1][j-1] + costFunction(string1,string2, i,j) => write chars under each other
            if self.semiglobalCostmatrix[i][j] == self.semiglobalCostmatrix[i - 1][j - 1] + self.costFunction(self.string1, self.string2, i, j):
                s1_out = self.string1[i - 1] + s1_out
                s2_out = self.string2[j - 1] + s2_out
                i -= 1
                j -= 1

            #entry D[i][j] resulted from D[i][j-1] + insertCost => add "-" (gap-character) to s1_out and add char from string2 to s2_out
            elif self.semiglobalCostmatrix[i][j] == self.semiglobalCostmatrix[i][j - 1] + self.insertCost:
                s1_out = "-" + s1_out
                s2_out = self.string2[j - 1] + s2_out
                j -= 1

            #entry D[i][j] resulted from D[i][j-1] + deleteCost => add "-" (gap-character) to s2_out and add char from string1 to s1_out
            else:
                s1_out = self.string1[i - 1] + s1_out
                s2_out = "-" + s2_out
                i -= 1
 
       #fill up s1_out with "-" (gap-character), so alignment strings are equal length
        while i > 0:
            s1_out = self.string1[i - 1] + s1_out
            s2_out = "-" + s2_out
            i -= 1

        #fill up s2_out with "-" (gap-character), so alignment strings are equal length
        while j > 0:
            s1_out = "-" + s1_out
            s2_out = self.string2[j - 1] + s2_out
            j -= 1

        return [s1_out, s2_out]

    #Returns two local alignment strings from lcoal cost matrix
    def getLocalTraceback(self):

        s1_out = ""
        s2_out = ""
        out = []

        #find indexes of max values in local cost matrix
        indexesMax = np.where(self.localCostmatrix == np.amax(self.localCostmatrix))

        #loop through all max values, each max value results in a different local alignemnt
        for ii in range(len(indexesMax[0])):
            
            #more than one max value
            if len(indexesMax[0]) >= 2:
                indexMax = indexesMax[ii]
            #one max value
            else:
                indexMax = indexesMax

            i = int(indexMax[0])
            j = int(indexMax[1])

            #variable to check for zero entry
            t = self.localCostmatrix[i][j]

            #build up output strings until we find a zero
            while t != 0:
                #entry D[i][j] resulted from D[i-1][j-1] + costFunction(string1,string2, i,j) => write chars under each other
                if self.localCostmatrix[i][j] == self.localCostmatrix[i - 1][j - 1] + self.ScoreFunction(self.string1,self.string2, i,j):
                    s1_out = self.string1[i - 1] + s1_out
                    s2_out = self.string2[j - 1] + s2_out
                    t = self.localCostmatrix[i - 1][j - 1]
                    i -= 1
                    j -= 1

                #entry D[i][j] resulted from D[i][j-1] + insertCost => add "-" (gap-character) to s1_out and add char from string2 to s2_out
                elif self.localCostmatrix[i][j] == self.localCostmatrix[i][j - 1] + self.insertCost:
                    s1_out = "-" + s1_out
                    s2_out = self.string2[j - 1] + s2_out
                    t = self.localCostmatrix[i][j - 1]
                    j -= 1

                #entry D[i][j] resulted from D[i][j-1] + deleteCost => add "-" (gap-character) to s2_out and add char from string1 to s1_out
                else:
                    s1_out = self.string1[i - 1] + s1_out
                    s2_out = "-" + s2_out
                    t = self.localCostmatrix[i-1][j]
                    i -= 1

            out.append(s1_out)
            out.append(s2_out)
            s1_out = ""
            s2_out = ""

        return out

    #Returns numpy global Costmatrix
    def getGlobalCostMatrix(self):
        l1 = len(self.string1) + 1
        l2 = len(self.string2) + 1
        # initializes Numpy Matrix with zeros
        D = np.zeros(shape=(l1, l2)).astype('int')

        #adds insertCost to every entry in first row
        for i in range(1, l2):
            X = D[0][i - 1] + self.insertCost
            D[0][i] = X

        #adds insertCost to every entry in first row
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


    #Returns numpy semi global Costmatrix
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
    #Returns numpy local Costmatrix
    def getLocalCostMatrix(self):
        l1 = len(self.string1) + 1
        l2 = len(self.string2) + 1
        # initializes Numpy Matrix with zeros
        D = np.zeros(shape=(l1, l2)).astype('int')

        # initializes first row with zeros because of local
        for i in range(1, l2):
            D[0][i] = 0


        # initializes first column with zeros because of local
        for j in range(1, l1):
            D[j][0] = 0


        # loops through all matrix entrys except D[0][0]
        # fills all entrtys according to alignment rules and custom Insertion-, Deletion- and Missmatch-Cost
        # calls function ScoreFunction(str1,str2,i,j)
        for i in range(1, l1):
            for j in range(1, l2):

                right = D[i][j - 1] + self.deleteCost
                
                #no entrys smaller 0
                if right < 0:
                    right = 0

                down = D[i - 1][j] + self.insertCost

                #no entrys smaller 0
                if down < 0:
                    down = 0

                diag = D[i - 1][j - 1] + self.ScoreFunction(self.string1, self.string2, i, j)
                
                #no entrys smaller 0
                if diag < 0:
                    diag = 0

                D[i][j] = max(right, down, diag)

        return D



















