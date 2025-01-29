from keras.models import Model
from keras.layers import (Input, Conv2D, MaxPooling2D, BatchNormalization, 
                          Dropout, add, ReLU, Softmax, Conv2DTranspose)

def encoder_block(x, input_filters: int):
     
     """
     Encoder block: 
         
     :x: Input tensor.
     :input_filters: Number of filters in Conv2D.
     :return: Processed tensor and skip connection for add() in the decoder.
     """
   
     # First convolution layer with dimensionality reduction (strides=2)
     x = Conv2D(input_filters, (3, 3), strides=2, use_bias=False, padding='same', kernel_initializer='he_normal')(x)
     x = BatchNormalization()(x)
     x = ReLU()(x)

     # Second convolution layer
     x = Conv2D(input_filters, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(x)
     x = BatchNormalization()(x)
     x = ReLU()(x)

     sc = x # First skip connection

     # Third convolution layer
     x = Conv2D(input_filters, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(x)
     x = BatchNormalization()(x)
     x = ReLU()(x)

     # Fourth convolution layer
     x = Conv2D(input_filters, (3, 3), use_bias=False, padding='same', kernel_initializer='he_normal')(x)
     x = BatchNormalization()(x)
     x = ReLU()(x)

     # Second skip connection
     x = add([x, sc])
     skip_connection = x

     return x, skip_connection
 
def decoder_block(x, skip_connection, input_filters, output_filters: int):
     
     """
     Decoder block:
         
     :x: Input tensor.
     :skip_connection: The corresponding feature map from the encoder
     :input_filters: Number of input filters (m)
     :output_filters: Number of output filters (n)
     :return: Processed tensor after the decoder block
     """
   
     # 1x1 convolution → reduction of filters by a factor of 4 (m/4)
     x = Conv2D(input_filters // 4, (1, 1), use_bias = False, padding='same', kernel_initializer='he_normal')(x)
     x = BatchNormalization()(x)
     x = ReLU()(x)
     
     # 3x3 transposed convolution → increase in spatial size
     x = Conv2DTranspose(input_filters // 4, (3,3), strides=2, use_bias = False, padding='same', kernel_initializer='he_normal' )(x)
     x = BatchNormalization()(x)
     x = ReLU()(x)
     
     x = Dropout(0.1)(x)
     
     # 1x1 convolution → increase the number of filters to output_filters
     x = Conv2D(output_filters, (1, 1), use_bias = False, padding='same', kernel_initializer='he_normal')(x)
     x = BatchNormalization()(x)
     x = ReLU()(x)
     
     # Add skip connection
     x = add([x, skip_connection])
     
     return x
 
def linknet_model(n_classes=4, IMG_HEIGHT=128, IMG_WIDTH=128, IMG_CHANNELS=1): #Don't forget to change n_classe if necessary
    
    """
    LinkNet: Semantic Segmentation 

    :n_classes: Number of classes in the segmentation task.
    :IMG_HEIGHT: Height of the input image.
    :IMG_WIDTH: Width of the input image.
    :IMG_CHANNELS: Number of channels (1 = grayscale, 3 = RGB).
    :return: Compiled Keras model.
    """

    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    #Initial block
    initial = Conv2D(64, (7,7), strides=2, use_bias = False, padding='same', kernel_initializer='he_normal')(inputs)
    initial = Dropout(0.1)(initial)
    initial = BatchNormalization()(initial)
    initial = ReLU()(initial)
    initial = MaxPooling2D((3,3), strides=2, padding='same')(initial)

    #Encoder
    e1, skip1 = encoder_block(initial, 64)
    e2, skip2 = encoder_block(e1, 128)
    e3, skip3 = encoder_block(e2, 256)
    e4, skip4 = encoder_block(e3, 512)

    #Decoder
    d4 = decoder_block(e4, skip3, 512, 256)
    d3 = decoder_block(d4, skip2, 256, 128)
    d2 = decoder_block(d3, skip1, 128, 64)
    d1 = decoder_block(d2, initial, 64, 64) # Connect to input
    
    #Output 
    out = Conv2DTranspose(32, (3,3), strides=2, use_bias = False, padding='same', kernel_initializer='he_normal')(d1)
    out = BatchNormalization()(out)
    out = ReLU()(out)
    
    out = Conv2D(32, (3,3), use_bias = False, padding='same', kernel_initializer='he_normal')(out)
    out = BatchNormalization()(out)
    out = ReLU()(out)
    
    out = Conv2DTranspose(n_classes, (2,2), strides=2, use_bias = False, padding='same', kernel_initializer='he_normal')(out)
    out = BatchNormalization()(out)
    outputs = Softmax()(out)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    
    return model

