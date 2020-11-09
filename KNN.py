def find(arr, i):
    newarr=sorted(arr[0:i+1])
    ind=newarr.index(arr[i])
    while(newarr[ind]==newarr[ind+1]):
        ind=ind+1

    return newarr[ind+1]

def maximumSum(a, m):
    modusum = []
    sum1 = 0
    for i in range(len(a)):
        sum1 = sum1 + a[i]
        modusum.append(sum1 % m)
    print(modusum)
    ans = []
    for i in range(1,len(modusum)):
        if (modusum[i ] > max(modusum[0:i])):

            continue
        else:
            print("iam"+str(i))
            newarr = modusum[0:i + 1]
            num1 = newarr[i]
            num = find(newarr,i)
            ans.append((num1 - num + m) % m)
            print(num1,num)
    print(ans)
    print("ans  "+str(max(ans)))

arr=[3,3,9,9,5]

maximumSum(arr,7)
