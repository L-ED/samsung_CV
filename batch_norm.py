'''
В данном шаге вам требуется реализовать функцию батч-нормализации без использования стандартной функции со следующими упрощениями:

Параметр Бета принимается равным 0.
Параметр Гамма принимается равным 1.
Функция должна корректно работать только на этапе обучения.
Вход имеет размерность число элементов в батче * длина каждого инстанса.
'''

import numpy as np
import torch
import torch.nn as nn

def custom_batch_norm1d(input_tensor, eps):
    sd = torch.mean(input_tensor,0,keepdim = True)
    var = torch.var(input_tensor, 0,keepdim = True,unbiased=False)
    normed_tensor =(input_tensor - sd)/((var+eps)**0.5)# Напишите в этом месте нормирование входного тензора
    return normed_tensor


input_tensor = torch.Tensor([[0.0, 0, 1, 0, 2], [0, 1, 1, 0, 10]])
batch_norm = nn.BatchNorm1d(input_tensor.shape[1], affine=False)

# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
# import numpy as np
# all_correct = True
# for eps_power in range(10):
#     eps = np.power(10., -eps_power)
#     batch_norm.eps = eps
#     batch_norm_out = batch_norm(input_tensor)
#     custom_batch_norm_out = custom_batch_norm1d(input_tensor, eps)

#     all_correct &= torch.allclose(batch_norm_out, custom_batch_norm_out)
#     all_correct &= batch_norm_out.shape == custom_batch_norm_out.shape
# print(all_correct)

'''
Немного обобщим функцию с предыдущего шага - добавим возможность задавать параметры Бета и Гамма.

На данном шаге вам требуется реализовать функцию батч-нормализации без использования стандартной функции со следующими упрощениями:

Функция должна корректно работать только на этапе обучения.
Вход имеет размерность число элементов в батче * длина каждого инстанса
'''

input_size = 7
batch_size = 5
input_tensor = torch.randn(batch_size, input_size, dtype=torch.float)

eps = 1e-3
batch_norm = nn.BatchNorm1d(input_size, eps=eps)
batch_norm.bias.data = torch.randn(input_size, dtype=torch.float)
batch_norm.weight.data = torch.randn(input_size, dtype=torch.float)


def custom_batch_norm1d(input_tensor, weight, bias, eps):
    sd = torch.mean(input_tensor,0,keepdim = True)
    var = torch.var(input_tensor, 0,keepdim = True,unbiased=False)
    normed_tensor =(input_tensor - sd)/((var+eps)**0.5)# Напишите в этом месте нормирование входного тензора
    return normed_tensor*weight + bias
    
# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
# batch_norm_out = batch_norm(input_tensor)
# custom_batch_norm_out = custom_batch_norm1d(input_tensor, batch_norm.weight.data, batch_norm.bias.data, eps)

# print(torch.allclose(batch_norm_out, custom_batch_norm_out) \
#       and batch_norm_out.shape == custom_batch_norm_out.shape)

'''
Избавимся еще от одного упрощения - реализуем работу слоя батч-нормализации на этапе предсказания.

На этом этапе вместо статистик по батчу будем использовать экспоненциально сглаженные статистики из истории обучения слоя.

В данном шаге вам требуется реализовать полноценный класс батч-нормализации без использования стандартной функции, 
принимающий на вход двумерный тензор. Осторожно, расчёт дисперсии ведётся по смещенной выборке, а расчет скользящего среднего по несмещенной.
'''

input_size = 3
batch_size = 5
eps = 1e-1


class CustomBatchNorm1d:
    def __init__(self, weight, bias, eps, momentum):
        
        self.weight =  weight
        self.bias = bias
        self.eps = eps
        self.momentum = momentum
        self.mu = 0
        self.var = 1
        self.mu_r = 0
        self.var_r = 1
        self.FLAG = 0
        
        # Реализуйте в этом месте конструктор.

    def __call__(self, input_tensor):
        if self.FLAG == 0:
            
            self.mu = torch.mean(input_tensor,0,keepdim = True)
            self.var = torch.var(input_tensor, 0,keepdim = True,unbiased= False)
            normed_tensor =(input_tensor - self.mu)/((self.var + self.eps)**0.5)
            
            self.var = torch.var(input_tensor, 0,keepdim = True,unbiased= True)
            self.mu_r = self.momentum*self.mu_r +(1 - self.momentum)*self.mu
            self.var_r = self.momentum*self.var_r +(1 - self.momentum)*self.var
        
        else:
            
            normed_tensor =(input_tensor - self.mu_r)/((self.var_r + self.eps)**0.5)
 
        return normed_tensor*self.weight + self.bias

    def eval(self):
        self.FLAG = 1
        # В этом методе реализуйте переключение в режим предикта.


batch_norm = nn.BatchNorm1d(input_size, eps=eps)
batch_norm.bias.data = torch.randn(input_size, dtype=torch.float)
batch_norm.weight.data = torch.randn(input_size, dtype=torch.float)
batch_norm.momentum = 0.5

custom_batch_norm1d = CustomBatchNorm1d(batch_norm.weight.data,
                                        batch_norm.bias.data, eps, batch_norm.momentum)

# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
# all_correct = True

# for i in range(8):
#     torch_input = torch.randn(batch_size, input_size, dtype=torch.float)
#     norm_output = batch_norm(torch_input)
#     custom_output = custom_batch_norm1d(torch_input)
#     all_correct &= torch.allclose(norm_output, custom_output) \
#         and norm_output.shape == custom_output.shape

# batch_norm.eval()
# custom_batch_norm1d.eval()

# for i in range(8):
#     torch_input = torch.randn(batch_size, input_size, dtype=torch.float)
#     norm_output = batch_norm(torch_input)
#     custom_output = custom_batch_norm1d(torch_input)
#     all_correct &= torch.allclose(norm_output, custom_output) \
#         and norm_output.shape == custom_output.shape
# print(all_correct)

'''
На данном шаге вам предлагается реализовать батч-норм слой для четырехмерного входа (например, батч из многоканальных двумерных картинок) без использования стандартной реализации со следующими упрощениями:

Параметр Бета = 0.
Параметр Гамма = 1.
Функция должна корректно работать только на этапе обучения.

'''

eps = 1e-3

input_channels = 3
batch_size = 3
height = 10
width = 10

batch_norm_2d = nn.BatchNorm2d(input_channels, affine=False, eps=eps)

input_tensor = torch.randn(batch_size, input_channels, height, width, dtype=torch.float)


def custom_batch_norm2d(input_tensor, eps):
    shapes = input_tensor.shape
    input_tensor = input_tensor.reshape(shapes[0], shapes[1], shapes[2]*shapes[3]).float()
    normed_tensor = torch.zeros(input_tensor.shape)
    
    for i in range(input_tensor.shape[1]):
        mean = torch.mean(input_tensor[:,i,:])
        var = torch.var(input_tensor[:,i,:],unbiased=False)
        
        normed_tensor[:,i,:] =  (input_tensor[:,i,:] - mean)/((var+eps)**0.5)
    return normed_tensor.reshape(shapes)


# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
# norm_output = batch_norm_2d(input_tensor)
# custom_output = custom_batch_norm2d(input_tensor, eps)
# print(torch.allclose(norm_output, custom_output) and norm_output.shape == custom_output.shape)

'''
