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
#takes two strings as input
#prints Matrix 
#returns Edit-distance ( D[|l1|][|l2|] )
def getCostMatrix(s1,s2):
	l1 = len(s1)+1
	l2 = len(s2)+1

	#initializes Numpy Matrix with zeros
	D = np.zeros(shape=(l1,l2)).astype('int')


	#initializes first row
	for i in range(1,l2):
		X = D[0][i-1] + insCost
		D[0][i] = X 


	#initializes first column
	for j in range(1,l1):
		X = D[j-1][0] + delCost
		D[j][0] = X


	#loops through all matrix entrys except D[0][0]
	#fills all entrtys according to alignment rules and custom Insertion-, Deletion- and Missmatch-Cost
	#calls function CostFunction(str1,str2,i,j)
	for i in range(1,l1):
		for j in range(1,l2):

			right = D[i-1][j] + delCost

			down = D[i][j-1] + insCost

			diag = D[i-1][j-1] + CostFunction(s1,s2,i,j)

			D[i][j] = min(right,down,diag)

	#print info, matrix and return edit-distance
	print("Cost-Matrix for: ")
	print("String 1: " + str(string1))
	print("String 2: " + str(string2))
	print()
	print(D)
	print()
	print()
	return D[l1-1][l2-1]





print("Edit-distance for String1 and String2: " + str(getCostMatrix(string1,string2)))







