numbers=[5,6,8,8,5]
for i in range(0, 3):
    numbers[i] = numbers[i] + 1
for j in range(3 + 1, len(numbers)):
    numbers[j] = numbers[j] + 1
print(numbers)