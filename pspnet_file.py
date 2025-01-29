from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D, Reshape, UpSampling2D, 
                          GlobalAveragePooling2D, concatenate, Conv2DTranspose, 
                          BatchNormalization, Dropout, Lambda, add, ReLU, Softmax, 
                          AveragePooling2D)

def encoder_block(x, input_filters: int):
     
     """
     Encoder block: 
         
     :x: Input tensor.
     :input_filters: Number of filters in Conv2D.
     :return: Processed tensor and skip connection for add() in the decoder.
     """
  
     #First skip connection
     shortcut_1 = Conv2D(input_filters, (1, 1), strides=2, use_bias=False, padding='same', kernel_initializer='he_normal')(x)
     shortcut_1 = BatchNormalization()(shortcut_1)
    
     # First convolution layer with dimensionality reduction (strides=2)
     x = Conv2D(input_filters, (3, 3), strides=2, use_bias=False, padding='same', kernel_initializer='he_normal')(x)
     x = BatchNormalization()(x)
     x = ReLU()(x)

     # Second convolution layer
     x = Conv2D(input_filters, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(x)
     x = BatchNormalization()(x)
     x = ReLU()(x)

     #Add function x and shortcut_1
     x = add([x, shortcut_1]) 
     #Second skip connection
     shortcut_2 = x


     # Third convolution layer
     x = Conv2D(input_filters, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(x)
     x = BatchNormalization()(x)
     x = ReLU()(x)

     # Fourth convolution layer
     x = Conv2D(input_filters, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(x)
     x = BatchNormalization()(x)
     x = ReLU()(x)

     #Add function x and shortcut_1
     x = add([x, shortcut_2])
     skip_connection = x #For the decoder

     return x, skip_connection

def red_ppm_block(x, sizes: int):

  # red_pixel (глобальный pooling => 1x1 => upsample до 32x32)
    x = GlobalAveragePooling2D()(x)   # B,H,W,C -> B,C
    x = Reshape((1, 1, -1))(x)         # -> B,1,1,C
    x = Conv2D(64, (1,1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = UpSampling2D(size=(sizes,sizes), interpolation='bilinear')(x)

    return x

def other_ppm_blocks(x, sizes: int):
   # yellow_pixel (pool=2x2 => 16x16 => 1x1 conv => upsample x2 => 32x32)
    x = AveragePooling2D(pool_size=(sizes, sizes))(x)  # -> 16x16
    x = Conv2D(64, (1,1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    x = UpSampling2D(size=(sizes, sizes), interpolation='bilinear')(x)
    # теперь 16x16 -> 32x32
    return x
    

def pspnet_model(n_classes=4, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):

  inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

  #Initial block
  initial = Conv2D(64, (7,7), strides=2, use_bias = False, padding='same', kernel_initializer='he_normal')(inputs)
  initial = Dropout(0.1)(initial)
  initial = BatchNormalization()(initial)
  initial = ReLU()(initial)
  initial = MaxPooling2D((3,3), strides=2, padding='same')(initial)
  
  #Encoder
  e1 = encoder_block(initial, 64)
  e2 = encoder_block(e1, 128)
  e3 = encoder_block(e2, 256)
  e4， = encoder_block(e3, 512)

  #Pyramid Pooling Module
  red_pixel = red_ppm_block(e4, 16)
  yellow_pixel = other_ppm_blocks(e4, 2)
  blue_pixel = other_ppm_blocks(e4, 4)
  green_pixel = other_ppm_blocks(e4, 8)

  common_result = concatenate([e4, red_pixel, yellow_pixel, blue_pixel, green_pixel])
  
  final_feature = UpSampling2D(size=(8,8), interpolation='bilinear')(common_result)

  outputs = Conv2D(n_classes, (3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(final_feature)
  outputs = BatchNormalization()(outputs)
  outputs = Softmax()(outputs)

  model = Model(inputs=[inputs], outputs=[outputs])
  return model


  
  
