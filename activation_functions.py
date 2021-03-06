'''
Проверим утверждение про затухание градиента на практике. В документации pytorch можно найти следующие функции активации (самые популярные мы подсветили жирным шрифтом.): 

ELU, Hardtanh, LeakyReLU, LogSigmoid, PReLU, ReLU, ReLU6, RReLU, SELU, CELU, Sigmoid, Softplus, Softshrink, Softsign, Tanh, Tanhshrink, Hardshrink.

Вам предстоит найти активацию, которая приводит к наименьшему затуханию градиента. 

Для проверки мы сконструируем SimpleNet, которая будет иметь внутри 3 fc-слоя, по 1 нейрону в каждом без bias'ов. Веса этих нейронов мы проинициализируем единицами. 
На вход в эту сеть будем подавать числа из нормального распределения. 
Сделаем 200 запусков (NUMBER_OF_EXPERIMENTS) для честного сравнения и посчитаем среднее значение градиента в первом слое. 
Найдите такую функцию, которая будет давать максимальные значения градиента в первом слое. 
Все функции активации нужно инициализировать с аргументами по умолчанию (пустыми скобками).
'''

import torch
import numpy as np

seed = int(input())
np.random.seed(seed)
torch.manual_seed(seed)

NUMBER_OF_EXPERIMENTS = 200

class SimpleNet(torch.nn.Module):
    def __init__(self, activation):
        super().__init__()

        self.activation = activation
        self.fc1 = torch.nn.Linear(1, 1, bias=False)  # one neuron without bias
        self.fc1.weight.data.fill_(1.)  # init weight with 1
        self.fc2 = torch.nn.Linear(1, 1, bias=False)
        self.fc2.weight.data.fill_(1.)
        self.fc3 = torch.nn.Linear(1, 1, bias=False)
        self.fc3.weight.data.fill_(1.)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        return x

    def get_fc1_grad_abs_value(self):
        return torch.abs(self.fc1.weight.grad)

def get_fc1_grad_abs_value(net, x):
    output = net.forward(x)
    output.backward()  # no loss function. Pretending that we want to minimize output
                       # In our case output is scalar, so we can calculate backward
    fc1_grad = net.get_fc1_grad_abs_value().item()
    net.zero_grad()
    return fc1_grad

activation =  torch.nn.Hardshrink()# Try different activations to get biggest gradient
              # ex.: torch.nn.Tanh()

net = SimpleNet(activation=activation)

fc1_grads = []
for x in torch.randn((NUMBER_OF_EXPERIMENTS, 1)):
    fc1_grads.append(get_fc1_grad_abs_value(net, x))

# Проверка осуществляется автоматически, вызовом функции:
# print(np.mean(fc1_grads))
# (раскомментируйте, если решаете задачу локально)
