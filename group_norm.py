'''
Нормализация "по группе" - это обобщение нормализации "по каналу" и "по инстансу".

Каналы в изображении не являются полностью независимыми, поэтому возможность использования статистики соседних каналов является преимуществом нормализации "по группе" по сравнению с нормализацией "по инстансу".

В то же время, каналы изображения могут сильно отличатся, поэтому нормализация "по группе" является более гибкой, чем нормализация "по каналу".

На этом шаге вам предлагается реализовать нормализацию "по группе" без использования стандартного слоя со следующими упрощениями:

Параметр Бета = 0.
Параметр Гамма = 1.
Требуется реализация только этапа обучения.
На вход подается трехмерный тензор.
Также слой принимает на вход число групп.

В слое нормализации "по группе" статистики считаются очень похоже на нормализацию "по каналу", только каналы разбиваются на группы.
'''
import torch
import torch.nn as nn

channel_count = 6
eps = 1e-3
batch_size = 20
input_size = 2

input_tensor = torch.randn(batch_size, channel_count, input_size)


def custom_group_norm(input_tensor, groups, eps):
    shapes = torch.tensor(input_tensor.shape)
    input_tensor = input_tensor.reshape(shapes[0], shapes[1], torch.prod(shapes[2:])).float()
    normed_tensor = torch.zeros(input_tensor.shape)
    group_len = input_tensor.shape[1]//groups
    for i in range(input_tensor.shape[0]):
        for j in range(groups):
            mean = torch.mean(input_tensor[i, j*group_len:(j+1)*group_len, :])
            var = torch.var(input_tensor[i, j*group_len:(j+1)*group_len, :],unbiased=False)

            normed_tensor[i, j*group_len:(j+1)*group_len, :] =  (input_tensor[i, j*group_len:(j+1)*group_len, :] - mean)/((var+eps)**0.5)
    return normed_tensor.reshape(shapes.tolist())


# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
# all_correct = True
# for groups in [1, 2, 3, 6]:
#     group_norm = nn.GroupNorm(groups, channel_count, eps=eps, affine=False)
#     norm_output = group_norm(input_tensor)
#     custom_output = custom_group_norm(input_tensor, groups, eps)
#     all_correct &= torch.allclose(norm_output, custom_output, 1e-3)
#     all_correct &= norm_output.shape == custom_output.shape
# print(all_correct)
