'''
Давайте начнем с разминки - реализуем функцию, добавляющую padding.

Пусть у нас есть батч input_images из двух изображений с тремя каналами (RGB). Размер изображений пусть будет 3*3. Вспомним, что вход сверточного слоя имеет следующую размерность:

размер батча

число каналов

высота

ширина

В рассматриваемом случае размерность входа (2, 3, 3, 3).

Если мы добавим вокруг каждого изображения отступ из одного нуля, то размер каждого изображений станет 3+2*1 = 5 пикселей в ширину и 5 в высоту соответственно (добавляем по одному нулю с каждой стороны изображения).

Напишите любую работающую реализацию.
'''

import torch

# Создаем входной массив из двух изображений RGB 3*3
input_images = torch.tensor(
      [[[[0,  1,  2],
         [3,  4,  5],
         [6,  7,  8]],

        [[9, 10, 11],
         [12, 13, 14],
         [15, 16, 17]],

        [[18, 19, 20],
         [21, 22, 23],
         [24, 25, 26]]],


       [[[27, 28, 29],
         [30, 31, 32],
         [33, 34, 35]],

        [[36, 37, 38],
         [39, 40, 41],
         [42, 43, 44]],

        [[45, 46, 47],
         [48, 49, 50],
         [51, 52, 53]]]])


def get_padding2d(input_images):
    padded_images = torch.nn.functional.pad(input_images, (1, 1, 1, 1) ).float()
    return padded_images


correct_padded_images = torch.tensor(
       [[[[0.,  0.,  0.,  0.,  0.],
          [0.,  0.,  1.,  2.,  0.],
          [0.,  3.,  4.,  5.,  0.],
          [0.,  6.,  7.,  8.,  0.],
          [0.,  0.,  0.,  0.,  0.]],

         [[0.,  0.,  0.,  0.,  0.],
          [0.,  9., 10., 11.,  0.],
          [0., 12., 13., 14.,  0.],
          [0., 15., 16., 17.,  0.],
          [0.,  0.,  0.,  0.,  0.]],

         [[0.,  0.,  0.,  0.,  0.],
          [0., 18., 19., 20.,  0.],
          [0., 21., 22., 23.,  0.],
          [0., 24., 25., 26.,  0.],
          [0.,  0.,  0.,  0.,  0.]]],


        [[[0.,  0.,  0.,  0.,  0.],
          [0., 27., 28., 29.,  0.],
          [0., 30., 31., 32.,  0.],
          [0., 33., 34., 35.,  0.],
          [0.,  0.,  0.,  0.,  0.]],

         [[0.,  0.,  0.,  0.,  0.],
          [0., 36., 37., 38.,  0.],
          [0., 39., 40., 41.,  0.],
          [0., 42., 43., 44.,  0.],
          [0.,  0.,  0.,  0.,  0.]],

         [[0.,  0.,  0.,  0.,  0.],
          [0., 45., 46., 47.,  0.],
          [0., 48., 49., 50.,  0.],
          [0., 51., 52., 53.,  0.],
          [0.,  0.,  0.,  0.,  0.]]]])

# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
# print(torch.allclose(get_padding2d(input_images), correct_padded_images))

'''
Попробуйте самостоятельно вывести формулу размерности выхода сверточного слоя, зная параметры входа и ядра. 
'''

import numpy as np


def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    out_shape = [input_matrix_shape[0], out_channels,((input_matrix_shape[2]+ 2*padding - kernel_size)//stride) +1, ((input_matrix_shape[3] + 2*padding - kernel_size)//stride) +1]# напишите тут код, вычисляющий выходную размерность

    return out_shape

print(np.array_equal(
    calc_out_shape(input_matrix_shape=[2, 3, 10, 10],
                   out_channels=10,
                   kernel_size=3,
                   stride=1,
                   padding=0),
    [2, 10, 8, 8]))

'''
На этом шаге требуется реализовать сверточный слой через циклы.

Обратите внимание, что в коде рассматривается общий случай - батч на входе не обязательно состоит из одного изображения, в ядре несколько слоев.
'''

from abc import ABC, abstractmethod


def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    batch_size, channels_count, input_height, input_width = input_matrix_shape
    output_height = (input_height + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    output_width = (input_width + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    return batch_size, out_channels, output_height, output_width


class ABCConv2d(ABC):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def set_kernel(self, kernel):
        self.kernel = kernel

    @abstractmethod
    def __call__(self, input_tensor):
        pass


class Conv2d(ABCConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride, padding=0, bias=False)

    def set_kernel(self, kernel):
        self.conv2d.weight.data = kernel

    def __call__(self, input_tensor):
        return self.conv2d(input_tensor)


def create_and_call_conv2d_layer(conv2d_layer_class, stride, kernel, input_matrix):
    out_channels = kernel.shape[0]
    in_channels = kernel.shape[1]
    kernel_size = kernel.shape[2]

    layer = conv2d_layer_class(in_channels, out_channels, kernel_size, stride)
    layer.set_kernel(kernel)

    return layer(input_matrix)


def test_conv2d_layer(conv2d_layer_class, batch_size=2,
                      input_height=4, input_width=4, stride=2):
    kernel = torch.tensor(
                      [[[[0., 1, 0],
                         [1,  2, 1],
                         [0,  1, 0]],

                        [[1, 2, 1],
                         [0, 3, 3],
                         [0, 1, 10]],

                        [[10, 11, 12],
                         [13, 14, 15],
                         [16, 17, 18]]]])

    in_channels = kernel.shape[1]

    input_tensor = torch.arange(0, batch_size * in_channels *
                                input_height * input_width,
                                out=torch.FloatTensor()) \
        .reshape(batch_size, in_channels, input_height, input_width)

    custom_conv2d_out = create_and_call_conv2d_layer(
        conv2d_layer_class, stride, kernel, input_tensor)
    conv2d_out = create_and_call_conv2d_layer(
        Conv2d, stride, kernel, input_tensor)

    return torch.allclose(custom_conv2d_out, conv2d_out) \
             and (custom_conv2d_out.shape == conv2d_out.shape)


# Сверточный слой через циклы.
class Conv2dLoop(ABCConv2d):
    def __call__(self, input_tensor):
        out_batch = []
        out_shape = calc_out_shape(input_tensor.shape, self.out_channels, self.kernel_size, self.stride, 0)
        ker_s = self.kernel_size
        for picture in input_tensor:
            one_pic = []
            for fltr in self.kernel:
                onefil = []
                for i in range(out_shape[-2]):
                    for j in range(out_shape[-1]):
                        onefil.append(picture[: ,i*self.stride: i*self.stride + ker_s, j*self.stride: j*self.stride + ker_s].mul(fltr).sum())
                one_pic.append(onefil)     
            out_batch.append(one_pic)         
        
        output_tensor = torch.tensor(out_batch).reshape(out_shape) # Напишите в этом месте вычисление свертки с использованием циклов.
        return output_tensor

# Корректность реализации определится в сравнии со стандартным слоем из pytorch.
# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
# print(test_conv2d_layer(Conv2dLoop))

'''
Реализация через циклы очень неэффективна по производительности. Есть целых два способа сделать то же самое с помощью матричного умножения. 

На этом шаге будет реализация первым из них.

 

Рассмотрим свертку одного одноканального изображения размером 4*4 пикселя (значения пикселей обозначены через X).

Сворачивать будем с ядром из одного фильтра размером 3*3, веса обозначены через W.

Для простоты примем stride = 1.

Тогда выход Y будет иметь размерность 1*1*2*2 (в данном случае на входе одно изображение - это первая единица в размерности, в ядре один фильтр - это вторая единица в размерности выхода).
То есть даже в самом общем случае мы за одно умножение матриц можем получить ответ.

Но рассчитанный таким способом выход не совпадает по размерности с выходом стандартного слоя из PyTorch - нужно изменить размерность.

Напоминание: во всех шагах этого урока мы считаем bias в сверточных слоях нулевым.

Вам осталось реализовать преобразование ядра в описанный выше формат.

Обратите внимание, что в коде рассматривается общий случай - вход состоит из нескольких многослойных изображений, в ядре несколько слоев.
'''


def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    batch_size, channels_count, input_height, input_width = input_matrix_shape
    output_height = (input_height + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    output_width = (input_width + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    return batch_size, out_channels, output_height, output_width


class ABCConv2d(ABC):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def set_kernel(self, kernel):
        self.kernel = kernel

    @abstractmethod
    def __call__(self, input_tensor):
        pass


class Conv2d(ABCConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride, padding=0, bias=False)

    def set_kernel(self, kernel):
        self.conv2d.weight.data = kernel

    def __call__(self, input_tensor):
        return self.conv2d(input_tensor)


def create_and_call_conv2d_layer(conv2d_layer_class, stride, kernel, input_matrix):
    out_channels = kernel.shape[0]
    in_channels = kernel.shape[1]
    kernel_size = kernel.shape[2]

    layer = conv2d_layer_class(in_channels, out_channels, kernel_size, stride)
    layer.set_kernel(kernel)

    return layer(input_matrix)


def test_conv2d_layer(conv2d_layer_class, batch_size=2,
                      input_height=4, input_width=4, stride=2):
    kernel = torch.tensor(
                      [[[[0., 1, 0],
                         [1,  2, 1],
                         [0,  1, 0]],

                        [[1, 2, 1],
                         [0, 3, 3],
                         [0, 1, 10]],

                        [[10, 11, 12],
                         [13, 14, 15],
                         [16, 17, 18]]]])

    in_channels = kernel.shape[1]

    input_tensor = torch.arange(0, batch_size * in_channels *
                                input_height * input_width,
                                out=torch.FloatTensor()) \
        .reshape(batch_size, in_channels, input_height, input_width)

    custom_conv2d_out = create_and_call_conv2d_layer(
        conv2d_layer_class, stride, kernel, input_tensor)
    conv2d_out = create_and_call_conv2d_layer(
        Conv2d, stride, kernel, input_tensor)

    return torch.allclose(custom_conv2d_out, conv2d_out) \
             and (custom_conv2d_out.shape == conv2d_out.shape)


class Conv2dMatrix(ABCConv2d):
    # Функция преобразование кернела в матрицу нужного вида.
    # Функция преобразование кернела в матрицу нужного вида.
    def _unsqueeze_kernel(self, torch_input, output_height, output_width):
        
        out_h, out_w = output_height, output_width
        kernel_unsqueezed = torch.tensor([])
        
        ker_shp = self.kernel.shape[-1]
        tens_w, tens_h = torch_input.shape[-1],torch_input.shape[-2]

        
        for filtr in self.kernel:
            one_fltr = torch.tensor([])

            for channel in filtr:
                one_chnl = torch.tensor([])
                chnltostr = torch.tensor([])
                for i in range(len(channel)):
                    if i < (len(channel) - 1):
                        chnltostr = torch.cat((chnltostr, channel[i], torch.zeros(tens_w - ker_shp)), 0)
                    else:
                        chnltostr = torch.cat((chnltostr, channel[i]), 0)

                for i in range(out_h):
                    for j in range(out_w):
                        one_oper = torch.tensor([])

                        zrs_before = torch.zeros((i*tens_w + j)*self.stride)
                        zrs_after = torch.zeros((tens_h - ker_shp - i*self.stride)*tens_w + tens_w - ker_shp - j*self.stride)

                        one_oper = torch.cat((zrs_before, chnltostr, zrs_after), 0)
                        one_chnl = torch.cat((one_chnl, one_oper), 0)
                        
                one_chnl = one_chnl.reshape(out_w*out_h, int(len(one_chnl)/(out_w*out_h)))

                one_fltr = torch.cat((one_fltr,one_chnl), 1)

            kernel_unsqueezed = torch.cat((kernel_unsqueezed,one_fltr),1)                        
                        
    
        return kernel_unsqueezed

    def __call__(self, torch_input):
        batch_size, out_channels, output_height, output_width\
            = calc_out_shape(
                input_matrix_shape=torch_input.shape,
                out_channels=self.kernel.shape[0],
                kernel_size=self.kernel.shape[2],
                stride=self.stride,
                padding=0)

        kernel_unsqueezed = self._unsqueeze_kernel(torch_input, output_height, output_width)
        result = kernel_unsqueezed @ torch_input.view((batch_size, -1)).permute(1, 0)
        return result.permute(1, 0).view((batch_size, self.out_channels,
                                          output_height, output_width))
# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
# print(test_conv2d_layer(Conv2dMatrix))

'''
На прошлом шаге W’ имеет много нулей. Это снижает эффективность метода.

На этом шаге будет реализация через матрицы другим, более эффективным способом.

Пусть в этот раз на входе батч из одного трехслойного (RGB) изображения размером 3*3.

Пусть ядро имеет 2 фильтра шириной и высотой 2 пикселя.

Тогда выход должен иметь размерность 1*2*2*2.

Пусть W - веса ядра, X - значения входной матрицы, Y - значения на выходе.
'''

def calc_out_shape(input_matrix_shape, out_channels, kernel_size, stride, padding):
    batch_size, channels_count, input_height, input_width = input_matrix_shape
    output_height = (input_height + 2 * padding - (kernel_size - 1) - 1) // stride + 1
    output_width = (input_width + 2 * padding - (kernel_size - 1) - 1) // stride + 1

    return batch_size, out_channels, output_height, output_width


class ABCConv2d(ABC):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

    def set_kernel(self, kernel):
        self.kernel = kernel

    @abstractmethod
    def __call__(self, input_tensor):
        pass


def create_and_call_conv2d_layer(conv2d_layer_class, stride, kernel, input_matrix):
    out_channels = kernel.shape[0]
    in_channels = kernel.shape[1]
    kernel_size = kernel.shape[2]

    layer = conv2d_layer_class(in_channels, out_channels, kernel_size, stride)
    layer.set_kernel(kernel)

    return layer(input_matrix)


class Conv2d(ABCConv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                      stride, padding=0, bias=False)

    def set_kernel(self, kernel):
        self.conv2d.weight.data = kernel

    def __call__(self, input_tensor):
        return self.conv2d(input_tensor)


def test_conv2d_layer(conv2d_layer_class, batch_size=2,
                      input_height=4, input_width=4, stride=2):
    kernel = torch.tensor(
                      [[[[0., 1, 0],
                         [1,  2, 1],
                         [0,  1, 0]],

                        [[1, 2, 1],
                         [0, 3, 3],
                         [0, 1, 10]],

                        [[10, 11, 12],
                         [13, 14, 15],
                         [16, 17, 18]]]])

    in_channels = kernel.shape[1]

    input_tensor = torch.arange(0, batch_size * in_channels *
                                input_height * input_width,
                                out=torch.FloatTensor()) \
        .reshape(batch_size, in_channels, input_height, input_width)

    custom_conv2d_out = create_and_call_conv2d_layer(
        conv2d_layer_class, stride, kernel, input_tensor)
    conv2d_out = create_and_call_conv2d_layer(
        Conv2d, stride, kernel, input_tensor)

    return torch.allclose(custom_conv2d_out, conv2d_out) \
             and (custom_conv2d_out.shape == conv2d_out.shape)


class Conv2dMatrixV2(ABCConv2d):
    # Функция преобразования кернела в нужный формат.
    def _convert_kernel(self):
        converted_kernel =[]
        
        for fil in self.kernel:
            converted_kernel.append(fil.flatten().tolist())
       
    
        return torch.tensor(converted_kernel)

    # Функция преобразования входа в нужный формат.
    def _convert_input(self, torch_input, output_height, output_width):
        stride = self.stride
        h_out, w_out = output_height, output_width
        ker_size = self.kernel_size
        out = torch.tensor([])
        for img in torch_input:
            one_img = torch.tensor([])
            for i in range(h_out):
                for j in range(w_out):
                    one_oper = img[:, i*stride: i*stride + ker_size,j*stride: j*stride + ker_size].flatten()
                    one_img = torch.cat((one_img, one_oper), 0)
            out = torch.cat((out, torch.transpose(one_img.reshape(h_out*w_out, len(one_img)//(h_out*w_out)),0,1)), 1)
        
        converted_input = out# Реализуйте преобразование входа.
        return converted_input

    def __call__(self, torch_input):
        batch_size, out_channels, output_height, output_width\
            = calc_out_shape(
                input_matrix_shape=torch_input.shape,
                out_channels=self.kernel.shape[0],
                kernel_size=self.kernel.shape[2],
                stride=self.stride,
                padding=0)

        converted_kernel = self._convert_kernel()
        converted_input = self._convert_input(torch_input, output_height, output_width)

        conv2d_out_alternative_matrix_v2 = converted_kernel @ converted_input
        return conv2d_out_alternative_matrix_v2.transpose(0, 1).view(torch_input.shape[0],
                                                     self.out_channels, output_height,
                                                     output_width).transpose(1, 3).transpose(2, 3)

# Проверка происходит автоматически вызовом следующего кода
# (раскомментируйте для самостоятельной проверки,
#  в коде для сдачи задания должно быть закомментировано):
