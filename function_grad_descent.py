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

# Перепишите пример, используя torch.optim.SGD

w = torch.tensor([[5., 10.], [1., 2.]], requires_grad=True)
alpha = 0.001
optimizer = torch.optim.SGD([w], alpha) # put our code here. Do not forget [] inside SGD constructor !!!!

for _ in range(500):
    # it's critical to calculate function inside the loop:
    function = (w + 7).log().log().prod()
    function.backward()
    optimizer.step()
    optimizer.zero_grad()
