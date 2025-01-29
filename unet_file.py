from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, ReLU, Softmax

def encoder_block(x, input_filters: int):

     """
     Encoder block: 
         
     :x: Input tensor.
     :input_filters: Number of filters in Conv2D.
     :return: Processed tensor and skip connection for concatenate() in the decoder.
     """
    
    #Contraction path
    x = Conv2D(input_filters, (3, 3),  kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Dropout(0.1)(x)
    
    x = Conv2D(input_filters, (3, 3),  kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    #Save information in skip connection before pooling
    skip_connection = x
    x = MaxPooling2D((2, 2))(x)
    
    return x, skip_connection

def decoder_block(x, skip_connection, input_filters: int):

      """
     Decoder block:
         
     :x: Input tensor.
     :skip_connection: The corresponding feature map from the encoder
     :input_filters: Number of input filters
     :return: Processed tensor
     """
  
    #Expansive path 
    x = Conv2DTranspose(input_filters, (2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    #Concatenation 
    x = concatenate([x, skip_connection])
    
    x = Conv2D(input_filters, (3, 3),  kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Dropout(0.2)(x)
    
    x = Conv2D(input_filters, (3, 3),  kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

def bottleneck(x, input_filters: int):

      """
     Bottleneck:
         
     :x: Input tensor.
     :input_filters: Number of input filters
     :return: Processed bottleneck tensor
     """
  
    x = Conv2D(input_filters, (3, 3),  kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Dropout(0.3)(x)
    
    x = Conv2D(input_filters, (3, 3), kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

def unet_model(n_classes=4, IMG_HEIGHT=128, IMG_WIDTH=128, IMG_CHANNELS=1): #Don't forget to change n_classes if necessary

     """
    U-Net: Semantic Segmentation 

    :n_classes: Number of classes in the segmentation task.
    :IMG_HEIGHT: Height of the input image.
    :IMG_WIDTH: Width of the input image.
    :IMG_CHANNELS: Number of channels (1 = grayscale, 3 = RGB).
    :return: Compiled U-Net model.
    """
  
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    #Encoder
    e1, skip1 = encoder_block(inputs, 16)
    e2, skip2 = encoder_block(e1, 32)
    e3, skip3 = encoder_block(e2, 64)
    e4, skip4 = encoder_block(e3, 128)
    
    #Bottleneck
    b = bottleneck(e4, 256)
    
    #Decoder
    d4 = decoder_block(b, skip4, 128)
    d3 = decoder_block(d4, skip3, 64)
    d2 = decoder_block(d3, skip2, 32)
    d1 = decoder_block(d2, skip1, 16)
    
    #Output
    outputs = Conv2D(n_classes, (1, 1))(d1)
    outputs = BatchNormalization()(outputs)
    outputs = Softmax()(outputs)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    
    return model
