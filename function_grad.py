'''
Реализуйте расчет градиента для функции f(w) = prod(ln(ln(w_{i,j} + 7))
 в точке w = [[5, 10], [1, 2]]w =[[5,10],[1,2]]

Подсказка: перемножить все значения функции можно с помощью метода .prod()
'''

import torch

w = torch.tensor([[5., 10.],
                  [1., 2.]], requires_grad = True)
    
function =  (torch.log(torch.log(w + 7))).prod()
function.backward()
