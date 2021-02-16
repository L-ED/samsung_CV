'''
Реализуйте градиентный спуск для той же функции f(w) = prod(ln(ln(w_{ij} + 7))

Пусть начальным приближением будет w{t=0} = [[5, 10], 
                                             [1, 2]]
шаг градиентного спуска alpha=0.001.

Чему будет равен w{t=500}?
'''

import torch

w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)
alpha = 0.001

for _ in range(500):

    # it's critical to calculate function inside the loop:
    
    function = (w + 7).log().log().prod()
    function.backward()
    w.data -=  alpha*w.grad# put our code here
    w.grad.zero_()# something is missing here!

print(w) 
