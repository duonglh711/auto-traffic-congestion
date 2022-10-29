from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import AveragePooling2D, Activation
from keras.layers.normalization import BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import model_from_json
from matplotlib import cm as CM

from tensorflow.keras.layers.experimental.preprocessing import Resizing

def BNet(input_shape=(None, None, 3)):
    def __make_branch(prev, filter, kernel, padding = 'same', dilated = 1, stride=1):
      x = Conv2D(filter, kernel_size=kernel, padding=padding, dilation_rate = dilated,strides=stride,activation='sigmoid', use_bias=False)(prev)
      return x
    
    input = Input(input_shape)


#     b0 = __make_branch(input, 5, 11, 'same', 1)
#     b0 = __make_branch(b0, 12, 6, 'same', 1)
    b1 = __make_branch(input, 12, 11, 'same', 1)
    b1 = __make_branch(b1, 20, 5, 'same', 1)
    b2 = __make_branch(input, 14, 7, 'same', 1)
    b2 = __make_branch(b2, 28, 4, 'same', 1)
    b3 = __make_branch(input, 18, 5, 'same', 1)
    b3 = __make_branch(b3, 36, 3, 'same', 1)
    b4 = __make_branch(input, 22, 3, 'same', 1)
    b4 = __make_branch(b4, 44, 2, 'same', 1)

    layer1 = Activation('relu')(Concatenate()([b1,b2,b3, b4]))

    layer1_ = __make_branch(layer1, 64, 3, 'same', 2)
    layer1 = MaxPooling2D((2,2))(layer1)
    layer3 = MaxPooling2D((2,2))(layer1)
    layer4 =(__make_branch(layer3, 60, 3, 'same', 2))
    layer5 = (__make_branch(layer4, 64, 3, 'same', 2))
    layer5 = MaxPooling2D((2,2))(layer5)
    layer5 = Activation('relu')(layer5)
    
    final = __make_branch(layer5, 32, 3, 'same', 2)
    final = __make_branch(final, 16, 3, 'same', 2)
    final = __make_branch(final, 1, 1, 'same', 1)
    final = Activation('relu')(final)

    out = final
    model = Model(input, out)

    return model

def BNetH(input_shape=(None, None, 3)):
    def __make_branch(prev, filter, kernel, padding='same', dilated=1, stride=1):
        x = Conv2D(filter, kernel_size=kernel, padding=padding, dilation_rate=dilated, strides=stride,
                   activation='sigmoid', use_bias=False)(prev)
        return x

    input = Input(input_shape)

    b1 = __make_branch(input, 12, 9, 'same', 1)
    b1 = __make_branch(b1, 20, 8, 'same', 1)
    # b1 = __make_branch(b1, 28, 3, 'same', 1)
    # b1 = __make_branch(b1, 28, 3, 'same', 1)
    b2 = __make_branch(input, 14, 7, 'same', 1)
    b2 = __make_branch(b2, 28, 6, 'same', 1)
    # b2 = __make_branch(b2, 36, 3, 'same', 1)
    # b2 = __make_branch(b2, 36, 3, 'same', 1)
    b3 = __make_branch(input, 18, 5, 'same', 1)
    b3 = __make_branch(b3, 36, 4, 'same', 1)
    # b3 = __make_branch(b3, 44, 3, 'same', 1)
    # b3 = __make_branch(b3, 44, 3, 'same', 1)
    b4 = __make_branch(input, 22, 3, 'same', 1)
    b4 = __make_branch(b4, 44, 2, 'same', 1)
    # b4 = __make_branch(b4, 56, 3, 'same', 1)
    # b4 = __make_branch(b4, 52, 2, 'same', 1)
    # b5 = __make_branch(input, 30, 2, 'same', 1)
    # b5 = __make_branch(b5, 56, 2, 'same', 1)

    layer1 = Activation('relu')(Concatenate()([b1, b2, b3, b4]))

    # layer1 = __make_branch(layer1, 56, 3, 'same', 1)
    # layer1 = __make_branch(layer1, 56, 3, 'same', 1)
    # layer1 = MaxPooling2D((2,2))(layer1)
    # layer2 = __make_branch(layer1, 64, 3, 'same', 2)
    # layer3 = __make_branch(layer2, 64, 3, 'same', 2)
    # layer1 =__make_branch(layer1, 64, 3, 'same', 1)
    layer3 = MaxPooling2D((4, 4))(layer1)
    layer4 = __make_branch(layer3, 64, 3, 'same', 2)
    layer5 = __make_branch(layer3, 96, 3, 'same', 2)
    layer5 = MaxPooling2D((2, 2))(layer5)
    layer5 = Activation('relu')(layer5)
    layer5 = __make_branch(layer5, 64, 3, 'same', 2)
    final = __make_branch(layer5, 1, 1, 'same', 1)
    final = Activation('relu')(final)

    out = final
    model = Model(input, out)

    return model



def BNetv3(input_shape=(None, None, 3)):
    def __make_branch(prev, filter, kernel, padding='same', dilated=1, stride=1):
        x = Conv2D(filter, kernel_size=kernel, padding=padding, dilation_rate=dilated, strides=stride,
                   activation='sigmoid', use_bias=False)(prev)
        return x

    input = Input(input_shape)

    b1 = __make_branch(input, 12, 9, 'same', 1)
    b1 = __make_branch(b1, 20, 6, 'same', 1)
    # b1 = __make_branch(b1, 44, 3, 'same', 1)
    # b1 = __make_branch(b1, 28, 3, 'same', 1)
    b2 = __make_branch(input, 14, 7, 'same', 1)
    b2 = __make_branch(b2, 28, 5, 'same', 1)
    # b2 = __make_branch(b2, 44, 3, 'same', 1)
    # b2 = __make_branch(b2, 36, 3, 'same', 1)
    b3 = __make_branch(input, 18, 5, 'same', 1)
    b3 = __make_branch(b3, 36, 4, 'same', 1)
    # b3 = __make_branch(b3, 44, 3, 'same', 1)
    # b3 = __make_branch(b3, 44, 3, 'same', 1)
    b4 = __make_branch(input, 22, 3, 'same', 1)
    b4 = __make_branch(b4, 44, 3, 'same', 1)
    # b4 = __make_branch(b4, 44, 3, 'same', 1)
    # b4 = __make_branch(b4, 52, 2, 'same', 1)
    # b5 = __make_branch(input, 30, 2, 'same', 1)
    # b5 = __make_branch(b5, 56, 2, 'same', 1)

    layer1 = Activation('relu')(Concatenate()([b1, b2, b3, b4]))

    # layer1 = __make_branch(layer1, 56, 3, 'same', 1)
    # layer1 = __make_branch(layer1, 56, 3, 'same', 1)
    # layer1 = MaxPooling2D((2,2))(layer1)
    # layer1 = __make_branch(layer1, 128, 3, 'same', 1)
    # # layer3 = __make_branch(layer2, 64, 3, 'same', 2)
    # layer1 =__make_branch(layer1, 128, 3, 'same', 1)
    # layer1 =__make_branch(layer1, 128, 3, 'same', 1)
    layer3 = MaxPooling2D((4, 4))(layer1)
    layer1 =__make_branch(layer1, 128, 3, 'same', 1)
    layer4 = __make_branch(layer3, 256, 3, 'same', 2)
    layer5 = __make_branch(layer3, 256, 3, 'same', 2)
    layer5 = MaxPooling2D((2, 2))(layer5)
    layer5 = Activation('relu')(layer5)
    layer5 = __make_branch(layer5, 128, 3, 'same', 2)
    layer5 = __make_branch(layer5, 64, 3, 'same', 2)
    final = __make_branch(layer5, 1, 1, 'same', 1)
    final = Activation('relu')(final)

    out = final
    model = Model(input, out)

    return model