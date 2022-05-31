from oop import alignment
import sys


def main():
    insCost = int(sys.argv[1])
    delCost = int(sys.argv[2])
    misCost = int(sys.argv[3])
    matchCost = 0
    String = sys.argv[4]
    AlignStrs = list(sys.argv[5:])


    alignment1 = alignment("", "", matchCost, insCost, delCost, misCost, 0, 0, "", String, AlignStrs)


    print("Strings for MSA-Profile:")
    for st in AlignStrs:
        print(st)
    print()
    print()


    print("MSA-Profile:")
    print()
    print(alignment1.msaProfileDF)
    print()
    print()

    print("Optimal MSA-Profile-Alignment with: " + String)
    print()
    print(alignment1.ProfAligProfDF)
    print()
    print("Optimal-MSA-Profile-Alignment-Cost: " + str(alignment1.seqProfileAlignmentCost) )
    print()
    print()

    print("Common MSA-Profile-Alignment:")
    print()
    print(alignment1.CommonProfileDF)




    

    




if __name__ == '__main__':
    main()
