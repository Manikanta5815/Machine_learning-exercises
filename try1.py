import math
import os
import random
import re
import sys

def beautifulPairs(A, B):
    lis = []
    for i in range(len(A)):
        num1 = A[i]
        for j in range(len(B)):
            if (num1 == B[j]):
                se = [i, j]
                lis.append(se)
    count = len(lis)
    lis1 = [0] * count
    for i in range(len(lis)):
        if (lis1[i] == 0):
            print(lis)
            for j in range(i + 1, len(lis)):
                if (lis[i][0] == lis[j][0]):
                    lis1[j] = 0
                    count = count - 1
    return count + 1


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(raw_input())

    A = map(int, raw_input().rstrip().split())

    B = map(int, raw_input().rstrip().split())

    result = beautifulPairs(A, B)

    fptr.write(str(result) + '\n')

    fptr.close()
