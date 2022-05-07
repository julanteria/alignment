from oop import alignment
import sys


def main():
    string1 = sys.argv[1]
    string2 = sys.argv[2]
    insCost = int(sys.argv[3])
    delCost = int(sys.argv[4])
    misCost = int(sys.argv[5])
    matchCost = int(sys.argv[6])
    aligmentType = str(sys.argv[7])

    alignment1 = alignment(string1, string2, matchCost, insCost, delCost, misCost, aligmentType)
    print(alignment1.Costmatrix)


if __name__ == '__main__':
    main()
