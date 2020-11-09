def maximumSum(a):
    lis = []
    for i in range(len(a)):

        cou = 0
        for j in range(len(a) - i):
            lis1 = (a[cou:cou + i+1])
            cou = cou + 1
            print(lis1)
            lis.append(sum(lis1))
maximumSum([3 ,3, 9, 9, 5])