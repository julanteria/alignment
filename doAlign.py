from oop import alignment
import sys


def main():
    insCost = int(sys.argv[1])
    delCost = int(sys.argv[2])
    misCost = int(sys.argv[3])
    matchCost = 0
    AlignStrs_ = list(sys.argv[4:])

    AlignStrs1 = []
    AlignStrs2 = []

    ind = 0
    for i in AlignStrs_:
        if i != "$":
            AlignStrs1.append(i)
            ind+=1
        else:
            break
    AlignStrs2 = AlignStrs_[ind+1:]


    alignment1 = alignment("", "", matchCost, insCost, delCost, misCost, 0, 0, "", "", AlignStrs1, AlignStrs2)

    print("Strings for Profile1: {}".format(AlignStrs1))
    print()
    print("Profile1:")
    print(alignment1.msaProfile1DF)
    print()
    print()

    print("Strings for Profile2: {}".format(AlignStrs2))
    print()
    print("Profile2:")
    print(alignment1.msaProfile2DF)
    print()
    print()

    print("Optimal Profile-Profile-Alignemt of Profile1 and Profile2 with Cost {}:".format(alignment1.Cost))
    print()
    print(alignment1.optProfProfAligDF)
    print()
    print()

    print("Common profile of optimal Profile-Profile-Alignemt:")
    print()
    print(alignment1.CommonProfProfDF)
    print()
    print()







    #print(alignment1.msaProfile2)
    #print(alignment1.msaProfile2DF)

   


if __name__ == '__main__':
    main()
