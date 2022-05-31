from oop2 import alignment
import sys


def main():
    string1 = sys.argv[1]
    string2 = sys.argv[2]
    matchCost = int(sys.argv[3])
    insCost = int(sys.argv[4])
    delCost = int(sys.argv[5])
    misCost = int(sys.argv[6])
    wg = int(sys.argv[7])
    ws = int(sys.argv[8])
    aligmentType = str(sys.argv[9])

    alignment1 = alignment(string1, string2, matchCost, insCost, delCost, misCost, wg, ws, aligmentType)

    



if __name__ == '__main__':
    main()
