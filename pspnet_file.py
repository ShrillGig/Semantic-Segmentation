from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, 
                          GlobalAveragePooling2D, concatenate, BatchNormalization, 
                          Dropout, Add, ReLU, Softmax, AveragePooling2D, Reshape)

def resnet18_residual_block(x, filters, stride=1):
    """
    Residual block
    
    :x: Input tensor.
    filters: Number of filters in Conv2D.
    :return: Processed tensor
    """
    shortcut = x  # Save data
    
    if stride != 1:  # If stride=2, then do downsample through `1x1 conv
        shortcut = Conv2D(filters, (1, 1), strides=stride, use_bias=False, padding='same', kernel_initializer='he_normal')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Conv2D(filters, (3, 3), strides=stride, padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(filters, (3, 3), padding='same', use_bias=False, kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    # Skip-connection
    x = Add()([x, shortcut])  
    x = ReLU()(x)

    return x

def pspnet_backbone(inputs):
    """ 
    Resnet-18
    
    :inputs: Input tensor
    """
    # Input Conv
    x = Conv2D(64, (7, 7), strides=2, use_bias=False, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((3, 3), strides=2, padding='same')(x)  

    # ResNet 
    x = resnet18_residual_block(x, 64, stride=1)  
    x = resnet18_residual_block(x, 64, stride=1)  

    x = resnet18_residual_block(x, 128, stride=2) 
    x = resnet18_residual_block(x, 128, stride=1) 

    x = resnet18_residual_block(x, 256, stride=2) 
    x = resnet18_residual_block(x, 256, stride=1) 

    x = resnet18_residual_block(x, 512, stride=2) 
    x = resnet18_residual_block(x, 512, stride=1) 

    return x

def pyramid_pooling_module(x):
    """ Pyramid Pooling Module (PPM) 
    
    :x: Input tensor
    """

    input_shape = x.shape[1:3]  

    # 1x1 Global Pooling
    red_pixel = GlobalAveragePooling2D()(x)
    red_pixel = Reshape((1, 1, -1))(red_pixel)
    red_pixel = Conv2D(64, (1, 1), padding='same', use_bias=False)(red_pixel)
    red_pixel = BatchNormalization()(red_pixel)
    red_pixel = UpSampling2D(size=input_shape, interpolation='bilinear')(red_pixel)  

    # 2x2 Pooling
    yellow_pixel = AveragePooling2D(pool_size=(2, 2))(x)
    yellow_pixel = Conv2D(64, (1, 1), padding='same', use_bias=False)(yellow_pixel)
    yellow_pixel = BatchNormalization()(yellow_pixel)
    yellow_pixel = UpSampling2D(size=2, interpolation='bilinear')(yellow_pixel)  

    # 3x3 Pooling
    blue_pixel = AveragePooling2D(pool_size=(3, 3))(x)
    blue_pixel = Conv2D(64, (1, 1), padding='same', use_bias=False)(blue_pixel)
    blue_pixel = BatchNormalization()(blue_pixel)
    blue_pixel = UpSampling2D(size=3, interpolation='bilinear')(blue_pixel) 

    # 6x6 Pooling
    green_pixel = AveragePooling2D(pool_size=(6, 6))(x)
    green_pixel = Conv2D(64, (1, 1), padding='same', use_bias=False)(green_pixel)
    green_pixel = BatchNormalization()(green_pixel)
    green_pixel = UpSampling2D(size=input_shape, interpolation='bilinear')(green_pixel)  

    #Concatenate the whole features 
    return concatenate([x, red_pixel, yellow_pixel, blue_pixel, green_pixel])

def pspnet_model(n_classes=44, IMG_HEIGHT=192, IMG_WIDTH=192, IMG_CHANNELS=1):
    """ 
    PSPNet: Semantic Segmentation
    
     :n_classes: Number of classes in the segmentation task.
    :IMG_HEIGHT: Height of the input image.
    :IMG_WIDTH: Width of the input image.
    :IMG_CHANNELS: Number of channels (1 = grayscale, 3 = RGB).
    :return: Compiled Keras model.
    """
  
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    #Backbone
    backbone_output = pspnet_backbone(inputs)

    # Pyramid Pooling Module (PPM)
    ppm = pyramid_pooling_module(encoder_output)

    # Final upsampling
    final_feature = UpSampling2D(size=(IMG_HEIGHT // ppm.shape[1], IMG_WIDTH // ppm.shape[2]), interpolation='bilinear')(ppm)
    outputs = Conv2D(n_classes, (3,3), padding='same', use_bias=False, kernel_initializer='he_normal')(final_feature)
    outputs = BatchNormalization()(outputs)
    outputs = Softmax()(outputs)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
