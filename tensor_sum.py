'''
Реализуйте при помощи pyTorch функцию, которая возвращает сумму (x.sum()) элементов тензора X, строго превышающих значение limit, которое является входным значением алгоритма.

Входная матрица: X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
'''

import torch

X = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
limit = int(input())

larger_than_limit_sum = (X[X>limit]).sum()

print(larger_than_limit_sum)
