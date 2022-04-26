import sys
import numpy as np

#command-line arguments 
string1 = sys.argv[1]
string2 = sys.argv[2]
insCost = int(sys.argv[3])
delCost = int(sys.argv[4])
misCost = int(sys.argv[5])


#uncommend this line to see full Matrix-print for long sequences
#np.set_printoptions(threshold=sys.maxsize)



#returns 0 if chars are the same
#returns custom missmatch costs if chars are different
def CostFunction(str1,str2,i,j):

    ch1 = str1[i-1]
    ch2 = str2[j-1]

    if ch1 == ch2:
        return 0

    if ch1 != ch2:
        return misCost




#function for Costmatrix D(i,j)
#takes two string as input
#prints Matrix 
#returns Edit-distance ( D[|l1|][|l2|] )
def getCostMatrix(s1,s2):
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



#function to get a optimal global alignemt
#NP-Cost-matrix as input
#two strings as output
def getAlignemt(D):

    #output string
    s1_out = ""
    s2_out = ""
    perfpath = ""

    #loop indices
    i = len(string1)
    j = len(string2)


    #loop through the matrix
    #check if insertion, deletion or comparison of the chars was the minimum
    #construct the alignment-strings based on the minimum
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



D = getCostMatrix(string1,string2)


print("Cost-Matrix for: ")
print("String 1: " + str(string1))
print("String 2: " + str(string2))
print()
print(D)
print()
print()



print("Edit-distance for String1 and String2: " + str(D[len(string1)][len(string2)]))
print()
print()



align = getAlignemt(D)

print("One optimal global Alignment for String1 and String2: ")
print()
print(align[0])
print(align[1])
