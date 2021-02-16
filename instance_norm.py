'''
На этом шаге вам предлагается реализовать нормализацию "по инстансу" без использования стандартного слоя со следующими упрощениями:

Параметр Бета = 0.
Параметр Гамма = 1.
На вход подается трехмерный тензор (размер батча, число каналов, длина каждого канала инстанса).
Требуется реализация только этапа обучения.
В слое нормализации "по инстансу" статистики считаются по последней размерности (по каждому входному каналу каждого входного примера).
'''

import torch
import torch.nn as nn

eps = 1e-3

batch_size = 5
input_channels = 2
input_length = 30

instance_norm = nn.InstanceNorm1d(input_channels, affine=False, eps=eps)

input_tensor = torch.randn(batch_size, input_channels, input_length, dtype=torch.float)


def custom_instance_norm1d(input_tensor, eps):
    shapes = torch.tensor(input_tensor.shape)
    input_tensor = input_tensor.reshape(shapes[0], shapes[1], torch.prod(shapes[2:])).float()
    normed_tensor = torch.zeros(input_tensor.shape)
    
    for i in range(input_tensor.shape[0]):
        for j in range(input_tensor.shape[1]):
            mean = torch.mean(input_tensor[i,j,:])
            var = torch.var(input_tensor[i,j,:],unbiased=False)
        
            normed_tensor[i,j,:] =  (input_tensor[i,j,:] - mean)/((var+eps)**0.5)
    return normed_tensor.reshape(shapes.tolist())


# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
# norm_output = instance_norm(input_tensor)
# custom_output = custom_instance_norm1d(input_tensor, eps)
# print(torch.allclose(norm_output, custom_output) and norm_output.shape == custom_output.shape)
