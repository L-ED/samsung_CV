'''
На этом шаге вам предлагается реализовать нормализацию "по каналу" без использования стандартного слоя со следующими упрощениями:

Параметр Бета = 0.
Параметр Гамма = 1.
Требуется реализация только этапа обучения.
Нормализация делается по всем размерностям входа, кроме нулевой.
Обратите внимание, что размерность входа на данном шаге не фиксирована.

Уточним, что в слое нормализации "по каналу" статистики считаются по всем размерностям, кроме нулевой.
'''

import torch
import torch.nn as nn


eps = 1e-10


def custom_layer_norm(input_tensor, eps):
    shapes = torch.tensor(input_tensor.shape)
    input_tensor = input_tensor.reshape(shapes[0], shapes[1], torch.prod(shapes[2:])).float()
    normed_tensor = torch.zeros(input_tensor.shape)
    
    for i in range(input_tensor.shape[0]):
        mean = torch.mean(input_tensor[i,:,:])
        var = torch.var(input_tensor[i,:,:],unbiased=False)
        
        normed_tensor[i,:,:] =  (input_tensor[i,:,:] - mean)/((var+eps)**0.5)
    return normed_tensor.reshape(shapes.tolist())


# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
# all_correct = True
# for dim_count in range(3, 9):
#     input_tensor = torch.randn(*list(range(3, dim_count + 2)), dtype=torch.float)
#     layer_norm = nn.LayerNorm(input_tensor.size()[1:], elementwise_affine=False, eps=eps)
# 
#     norm_output = layer_norm(input_tensor)
#     custom_output = custom_layer_norm(input_tensor, eps)

#     all_correct &= torch.allclose(norm_output, custom_output, 1e-2)
#     all_correct &= norm_output.shape == custom_output.shape
# print(all_correct)

'''
