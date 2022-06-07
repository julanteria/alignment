import numpy as np
import pandas as pd
import random as rd
from fractions import Fraction as f
import sys

log = open('log.txt', 'a')




class alignment:

    # alignment attributes
    def __init__(self, string1, string2, ma, i, d, mi, wg, ws, aligmentType, String, AlignStrs1, AlignStrs2):
        #alignment1 = alignment("", "", matchCost, insCost, delCost, misCost, 0, 0, "", String, AlignStrs)
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

        self.multipleSequenceAlignment1 = AlignStrs1
        self.multipleSequenceAlignment2 = AlignStrs2

        self.alphabet = self.getAlphabet() #"H", "S", "A", "L", "-"]

       
        


        self.msaProfile1 = self.getMsaProfile(self.multipleSequenceAlignment1)[0]
        self.msaProfile1DF = self.getMsaProfile(self.multipleSequenceAlignment1)[1]

        self.msaProfile2 = self.getMsaProfile(self.multipleSequenceAlignment2)[0]
        self.msaProfile2DF = self.getMsaProfile(self.multipleSequenceAlignment2)[1]

        self.profProfCostMatrix = self.getCostMatrixProfileProfile()

        self.optProfProfAligDF = self.getProfProfAligTraceback()[0]
        self.optProfProfAlig1 = self.getProfProfAligTraceback()[1]
        self.optProfProfAlig2 = self.getProfProfAligTraceback()[2]

        #self.getCommonProfProf = self.getCommonProfProf()[0]
        self.CommonProfProfDF = self.getCommonProfProf()[1]





        #self.costMatrixSeqProfileAlignment = self.getCostMatrixSeqProfileAlignment()

        #self.ProfAligProf = self.getProfAligStrTraceback()[1] #opt?
        #self.ProfAligProfDF = self.getProfAligStrTraceback()[2]
        #self.ProfilAligntString = self.getProfAligStrTraceback()[0]

        #self.seqProfileAlignmentCost = self.getSeqProfAligStringCost()

        #self.CommonProfile = self.getCommonProfile()[0]
        #self.CommonProfileDF = self.getCommonProfile()[1]


    

        self.globalAlignment = []
        self.semiglobalAlignment = []
        self.localAlignment = []
        self.Cost = 1.25
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

    # alignment score function for minimazation optimization1
    def costFunction(self, str1, str2, i, j, ):

        ch1 = str1[i - 1]
        ch2 = str2[j - 1]

        if ch1 == ch2:
            return self.matchCost

        if ch1 != ch2:
            return self.missCost

        # alignment score function for minimazation optimization1
   
    def costFunctionProfileCol(self, msaProf, col, ch1):
        cost = 0
        ch1 = ch1
        i = 0
        j = col
        for ch2 in self.alphabet:
            if ch1 == ch2:
                cost += self.matchCost * msaProf[i][j]
            elif ch1 != ch2:
                if ch1 == "-" or ch2 == "-":
                    cost += self.insertCost * msaProf[i][j]
                else:
                    cost += self.missCost * msaProf[i][j]
            i+=1
        return cost

    def costFunctionProfileCol2(self, msaProf, msaProf2, col, ch1):
        cost = 0
        ch1 = ch1
        i = 0
        j = col
        for ch2 in self.alphabet:
            if ch1 == ch2:
                cost += self.matchCost * msaProf[i][j] * msaProf2[i][j]
            elif ch1 != ch2:
                if ch1 == "-" or ch2 == "-":
                    cost += self.insertCost * msaProf[i][j] * msaProf2[i][j]
                else:
                    cost += self.missCost * msaProf[i][j] * msaProf2[i][j]
            i+=1
        return cost






    # alignment score function for maximization optimization
    def ScoreFunction(self, str1, str2, i, j):

        ch1 = str1[i - 1]
        ch2 = str2[j - 1]

        if ch1 == ch2:
            return self.matchCost

        if ch1 != ch2:
            return self.missCost


    def getSeqProfAligStringCost(self):
        cost = 0
        j = 0

        for ch1 in self.seqProfileAlignmentString:
            i = 0
            for ch2 in self.alphabet:
                if ch1 == ch2:
                    cost += self.matchCost * self.ProfAligProf[i][j]
                elif ch1 != ch2:
                    if ch1 == "-" or ch2 == "-":
                        cost += self.insertCost * self.ProfAligProf[i][j]
                    else:
                        cost += self.missCost * self.ProfAligProf[i][j]
                i+=1
            j+=1

        return cost




    @staticmethod
    def costumMin(list):
        x = []
        for i in list:
            if i >= 0:
                x.append(i)

        return min(x)

    def appendIt(self, P):
        appender = []
        r = P.shape[0]-1
        for x in range(0,r):
            appender.append([0])
        appender.append([1])

        P = np.append(P, appender, axis=1) 
        return P


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


    def getAlphabet1(self):
        l = []
        for string in self.multipleSequenceAlignment1:
            l += list(string)
        charSet = set(l)
        out = list(charSet)
        if "-" in out:
            out.remove("-")
            out.append("-")


        return out

    def getAlphabet2(self):
        l = []
        for string in self.multipleSequenceAlignment2:
            l += list(string)
        charSet = set(l)
        out = list(charSet)
        if "-" in out:
            out.remove("-")
            out.append("-")

        return out 



    def getAlphabet(self):
        l = []
        for string in self.multipleSequenceAlignment1 + self.multipleSequenceAlignment2:
            l += list(string)
        charSet = set(l)
        out = list(charSet)
        if "-" in out:
            out.remove("-")
            out.append("-")

        return out 

    def getMsaProfile(self, stringList):
        charSet = self.alphabet
        
        alphLen = len(charSet)
        strLen = len(stringList[0])

        profile =  np.empty(shape=(alphLen,strLen), dtype='object')

        l = self.alphabet


        for i in range(alphLen):
            for j in range(strLen):
                char = l[i]
                x = 0
                for string in stringList:

                    if string[j] == char:
                        x += 1

                if x != 0:
                    profile[i][j] = f(x/len(stringList)).limit_denominator(10000) 
                else:
                    profile[i][j] = 0.0  


        


        df = pd.DataFrame(profile, index=l, columns=[x for x in range(1,strLen+1)])
        return profile, df
        



    def getCostMatrixSeqProfileAlignment(self):
        string_out = ""
        P_out = np.copy(self.msaProfile)
        appender = []
        r = P_out.shape[0]-1
        for x in range(0,r):
            appender.append([0])
        appender.append([1])



        m = len(self.multipleSequenceAlignment[0])+1  
        n = len(self.seqProfileAlignmentString)+1 
 
        D = np.zeros(shape=(n, m)).astype('object') 
    
        # initializes first row
        for j in range(1, m):

            X = D[0][j - 1] + self.costFunctionProfileCol(j-1, "-")
            D[0][j] = X

        # initializes first column
        for i in range(1, n):
            ch1 = "-"
            ch2 = self.seqProfileAlignmentString[i-1]
            cost = 0
            if ch1 != ch2:
                cost = self.missCost
            X = D[i-1][0] + cost
            D[i][0] = X
        index = 0

        for i in range(1, n):
            for j in range(1, m):
                ch1 = "-"
                ch2 = self.seqProfileAlignmentString[i-1]
                cost = 0
                if ch1 != ch2:
                    cost = self.missCost

                up = D[i-1][j] + cost

                left = D[i][j-1] + self.costFunctionProfileCol(j-1, "-")

                diag = D[i-1][j-1] + self.costFunctionProfileCol(j-1, ch2)


                mini = min(up,left,diag)
                D[i][j]= mini
        
        return D



    
    def getProfAligStrTraceback(self):
        raus = ""
        s1_out = ""
        P_out = np.copy(self.msaProfile)

        appender = []
        r = P_out.shape[0]-1
        for x in range(0,r):
            appender.append([0])
        appender.append([1])


        D = self.costMatrixSeqProfileAlignment
        shape = D.shape

        string = self.seqProfileAlignmentString


        
        i = shape[0]-1
        j = shape[1]-1

        while i > 0 and j > 0:

            if D[i][j] == D[i][j - 1] + self.costFunctionProfileCol(j-1, "-"):
                s1_out += "l"
                j -=1
            
            
            elif D[i][j] == D[i - 1][j - 1] + self.costFunctionProfileCol(j-1, string[i-1]):
                s1_out += "D"
                i -= 1
                j -=1
            

            else:
                np.insert(P_out, i, appender, axis=1)
                s1_out += "d"
                i -= 1
        
        x = 0
        for a in s1_out:
            if a == "l":
                raus += "-"
            elif a == "d":
                raus += string[x]
                np.insert(P_out, i, appender, axis=1)
                x += 1
            elif a == "D":
                raus += string[x]
                x += 1
        
        

        while P_out.shape[1] < len(raus):
            P_out = self.appendIt(P_out)

        while(P_out.shape[1] > len(raus)):
            i = rd.choice(list(range(0,len(raus))))
            raus = raus[:i] + "-" + raus[i:]
 
        


        df = pd.DataFrame(P_out, index=list(self.alphabet), columns=list(raus))

        return raus, P_out, df



    def getCommonProfile(self):
        stringAlig = self.ProfilAligntString
        

        B = np.empty(shape=(self.msaProfile.shape), dtype='object')

        shape = B.shape
        l = self.alphabet
        

        charSet = self.alphabet
        
        alphLen = len(charSet)
        strLen = len(self.multipleSequenceAlignment[0])


        for i in range(alphLen):
            for j in range(strLen):
                dow = len(self.multipleSequenceAlignment)
                char = l[i]
                x = 0
                for string in self.multipleSequenceAlignment:

                    if string[j] == char:
                        x += 1

                    B[i][j] = (x,dow)

                if stringAlig[j] == l[i]:
                    x += 1
                    dow += 1

                else:
                    dow += 1

                entry = 0

                if x == 0:
                    entry = 0
                else:
                    entry = f(x/dow)

                B[i][j] = entry




        df = pd.DataFrame(B, index=list(self.alphabet), columns=list(range(1,len(string)+1)))
        
        return B, df





    def getCostMatrixProfileProfile(self):
            string_out = ""
            P_out = np.copy(self.msaProfile1)



            m = len(self.multipleSequenceAlignment2[0])+1  
            n = len(self.multipleSequenceAlignment1[0])+1  
    
            D = np.zeros(shape=(n, m)).astype('float') 

        

            # initializes first row
            for j in range(1, m):
                X = D[0][j - 1] + self.costFunctionProfileCol(self.msaProfile2, j-1, "-")
                D[0][j] = X


            for i in range(1, n):
                X = D[i-1][0] + self.costFunctionProfileCol(self.msaProfile1, i-1, "-")
                D[i][0] = X



            alph1 = self.alphabet
            #alph2 = self.getAlphabet2()

            for i in range(1, n):
                for j in range(1, m):


                    #alphabet des jeweiligen MSA?
                    sumUP = 0
                    index = 0
                    cost = 0
                    sumUP = self.costFunctionProfileCol(self.msaProfile1, i-1, "-")


                    up = D[i-1][j] + sumUP

                    #alphabet des jeweiligen MSA?
                    sumLeft = 0
                    index = 0
                    cost = 0

                    sumLeft = self.costFunctionProfileCol(self.msaProfile2, j-1, "-")



                    left = D[i][j-1] + sumLeft


                    #alphabet des jeweiligen MSA?
                    sumDiag = 0
                    indexi = 0
                    indexj = 0
                    cost = 0
                
                    for chara in alph1:
                        indexi = 0   
                        for charb in alph1:
                            cost = 0
                            
                            if chara != charb:
                                cost = self.missCost

                            s = cost * self.msaProfile1[indexj][j-1] * self.msaProfile2[indexi][j-1]
                            sumDiag += s

                    

                            indexi+=1
                        indexj+=1




                    diag = D[i-1][j-1] + sumDiag

                    mini = min(up,left,diag)
                    D[i][j]= mini
            
            
            df = pd.DataFrame(D)
            return D

        
    def getProfProfAligTraceback(self):
        s1_out=""
        P1_out = np.copy(self.msaProfile1)
        P2_out = np.copy(self.msaProfile2)

        appender1 = []
        r = P1_out.shape[0]-1
        for x in range(0,r):
            appender1.append(0)
        appender1.append(1)

        appender2 = []
        r = P2_out.shape[0]-1
        for x in range(0,r):
            appender2.append(0)
        appender2.append(1)


        D = self.profProfCostMatrix
        
        shape = D.shape
        
        i = shape[0]-1
        j = shape[1]-1

        while i > 0 and j > 0:

            sumDiag = 0
            indexi = 0
            indexj = 0
            cost = 0
            for chara in self.alphabet:
                indexi = 0   
                for charb in self.alphabet:
                    cost = 0
                    if chara != charb:
                        cost = self.missCost

                    s = cost * self.msaProfile1[indexj][j-1] * self.msaProfile2[indexi][j-1]
                    sumDiag += s

                    indexi+=1
                indexj+=1

            if D[i][j] == D[i][j - 1] + self.costFunctionProfileCol(self.msaProfile2, j-1, "-"):
                s1_out += "l"
                j -=1

            
            
            elif D[i][j] == D[i - 1][j - 1] + sumDiag:
                s1_out += "D"
                i -= 1
                j -=1
            

            else:
                s1_out += "d"
                i -= 1
        
        x = 0
      
        for a in s1_out:
            if a == "l":
                P1_out = np.insert(P1_out, x, appender1, axis=1)
            elif a == "d":
                P2_out = np.insert(P2_out, x, appender2, axis=1)
                x += 1
            elif a == "D":
                x += 1





        B_out = np.concatenate((P1_out, P2_out), axis=0)

        B_outDF = (pd.DataFrame(B_out,index=self.alphabet + self.alphabet))

        return B_outDF, P1_out, P2_out

    def getCommonProfProf(self):
        numStr1 = len(self.multipleSequenceAlignment1)
        numStr2 = len(self.multipleSequenceAlignment2)
        numAll = numStr1 + numStr2

        B1 = self.optProfProfAlig1
        B2 = self.optProfProfAlig2
        B_out = np.copy(B1)

        shape = B1.shape

        n = shape[0]
        m = shape[1]


        for i in range(m):
            for j in range(n):
                sum = 0
                sum += B1[j][i] * numStr1
                sum += B2[j][i] * numStr2
                B_out[j][i] = f(sum/numAll).limit_denominator(10000) 




        df = pd.DataFrame(B_out, index=self.alphabet)


        return B_out, df

        