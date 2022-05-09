from oop import alignment
import sys


def main():
    string1 = sys.argv[1]
    string2 = sys.argv[2]
    matchCost = int(sys.argv[3])
    insCost = int(sys.argv[4])
    delCost = int(sys.argv[5])
    misCost = int(sys.argv[6])
    aligmentType = str(sys.argv[7])

    alignment1 = alignment(string1, string2, matchCost, insCost, delCost, misCost, aligmentType)
    

    if len(alignment1.globalAlignment) != 0:
        print()
        print(alignment1.globalCostmatrix)
        print(alignment1.globalAlignment[0])
        print(alignment1.globalAlignment[1])
        print()
    
    if len(alignment1.semiglobalAlignment) != 0:
        print()
        print(alignment1.semiglobalCostmatrix)
        print(alignment1.semiglobalAlignment[0])
        print(alignment1.semiglobalAlignment[1])
        print()


    if len(alignment1.localAlignment) != 0:
        print()
        print(alignment1.localCostmatrix)
        for i in range(0,len(alignment1.localAlignment),2):
            print()
            print(alignment1.localAlignment[i])
            print(alignment1.localAlignment[i+1])
            print()

if __name__ == '__main__':
    main()
