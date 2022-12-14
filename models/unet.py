import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization


def doubleConvBlock(out_channels, name=None):
    if name is not None:
        block = tf.keras.Sequential(name=name)
    else:
        block = tf.keras.Sequential()
    block.add(Conv2D(filters=out_channels, kernel_size=(
        3, 3), padding='same'))
    block.add(ReLU())
    block.add(Conv2D(filters=out_channels, kernel_size=(
        3, 3), padding='same'))
    block.add(ReLU())
    return block


def downBlock(out_channels, name="downBlock"):
    block = tf.keras.Sequential(name=name)
    block.add(MaxPool2D(pool_size=(2, 2)))
    block.add(doubleConvBlock(out_channels, name=name+"_doub_conv"))
    return block


def bridgeDown(out_channels, name="bridgeDown"):
    block = tf.keras.Sequential(name=name)
    block.add(MaxPool2D(pool_size=(2, 2)))
    block.add(Conv2D(filters=out_channels, kernel_size=(3, 3), padding='same'))
    block.add(ReLU())
    return block


def bridgeUp(out_channels, name="bridgeDown"):
    block = tf.keras.Sequential(name=name)
    block.add(Conv2D(filters=out_channels, kernel_size=(3, 3), padding='same'))
    block.add(ReLU())
    block.add(Conv2DTranspose(filters=out_channels,
                              kernel_size=(2, 2), strides=(2, 2), padding='same'))
    return block

# Returns output, not keras.model instance


def upBlock(x1, x2, out_channels):
    x = Concatenate()([x1, x2])
    conv1 = doubleConvBlock(x1.shape[-1])(x)
    cvt = Conv2DTranspose(out_channels, kernel_size=(
        2, 2), strides=(2, 2))(conv1)
    return ReLU()(cvt)

# Returns output, not keras.model instance


def outputBlock(x1, x2, out_channels):
    x = Concatenate()([x1, x2])
    x = doubleConvBlock(out_channels)(x)
    x = Conv2D(filters=out_channels, kernel_size=(1, 1), padding='same')(x)
    return x


class DeepWBnet():
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def get_model(self):
        inputs = Input(shape=self.input_shape, name="Main Input")
        encoder_inc = doubleConvBlock(24, name="encoder_inc")(inputs)
        encoder_down1 = downBlock(48, name="encoder_down1")(encoder_inc)
        encoder_down2 = downBlock(96, name="encoder_down2")(encoder_down1)
        encoder_down3 = downBlock(192, name="encoder_down3")(encoder_down2)
        encoder_bridge_down = bridgeDown(
            384, name="encoder_bridge_down")(encoder_down3)
        decoder_bridge_up = bridgeUp(
            192, name="decoder_bridge_up")(encoder_bridge_down)
        decoder_up1 = upBlock(decoder_bridge_up, encoder_down3, 96)
        decoder_up2 = upBlock(decoder_up1, encoder_down2, 48)
        decoder_up3 = upBlock(decoder_up2, encoder_down1, 24)
        outputs = outputBlock(decoder_up3, encoder_inc, 1)
        model = tf.keras.Model(inputs, outputs)
        return model


def ConvBNRelu(filters=64, kernel_size=(4, 4), stride=(2, 2), padding="same", init=RandomNormal(stddev=0.2), batch_norm=True):
    """
    A custom layer for Convolution, Batch Norm and ReLu activation. 
    This combination repeats multiple times in the model definition.
    Parameters 
    ----------
    filters : Convolutional layer number of filters
        DESCRIPTION. The default is 64.
    kernel_size : Convolutional layer kernel size
        DESCRIPTION. The default is (4, 4).
    stride : Convolutional layer stride
        DESCRIPTION. The default is (2, 2).
    padding : Convolutional layer padding
        DESCRIPTION. The default is "same".
    init : parameter initializer
        DESCRIPTION. The default is RandomNormal(stddev=0.2).
    batch_norm : should it have a batch norm layer
        DESCRIPTION. The default is True.
    Returns
    -------
    block : Activation output
    """
    block = tf.keras.Sequential()
    block.add(Conv2D(filters, kernel_size, strides=stride,
                     padding=padding, kernel_initializer=init))
    if batch_norm:
        block.add(BatchNormalization())
    block.add(LeakyReLU(alpha=0.2))
    return block


def decoder_block(inputs, skip_inputs, filters, dropout=True, padding="same"):
    """
    Decoder block for the decoder in the U-Net of the generator.
    Parameters
    ----------
    inputs : Input from the previous layer.
        DESCRIPTION. Functional input from previous layer.
    skip_inputs : Residual skip connections from encoder blocks. 
        DESCRIPTION.
    filters : number of filters convolutional layer
        DESCRIPTION.
    dropout : dropout boolean to have it or not.
        DESCRIPTION. The default is True.
    Returns
    -------
    g : activation output after forward pass of the block.
        DESCRIPTION.
    """
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(filters, (4, 4), strides=(
        2, 2), padding=padding, kernel_initializer=init)(inputs)
    if dropout:
        g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, skip_inputs])
    g = Activation('relu')(g)
    return g


class Generator():
    def __init__(self, input_shape):
        """
        Generator constructor.
        Parameters
        ----------
        input_shape : requires input shape for static graph building.
            DESCRIPTION.
        Returns
        -------
        None.
        """
        self.input_shape = input_shape
        self.init = RandomNormal(stddev=0.2)

    def get_model(self):
        """
        Constructs the static graph and returns the generator model object.
        Returns
        -------
        model : Generator object
            DESCRIPTION.
        """
        # Down Path
        inputs = Input(shape=self.input_shape)
        e1 = ConvBNRelu(filters=64, batch_norm=False)(inputs) 
        e2 = ConvBNRelu(filters=128)(e1) 
        e3 = ConvBNRelu(filters=256)(e2) 
        e4 = ConvBNRelu(filters=512, padding="valid")(e3)
        e5 = ConvBNRelu(filters=512)(e4)
        e6 = ConvBNRelu(filters=512, padding="valid")(e5)
        
        # Bottle Neck Layers
        bottle_neck1 = ConvBNRelu(filters=512)(e6) 
        bottle_neck2 = Conv2D(512, (4, 4), strides=(
            1, 1), padding='same', kernel_initializer=self.init)(bottle_neck1) 
        a = Activation('relu')(bottle_neck2) 
        
        # Up Path
        d1 = decoder_block(a, e6, 512) 
        d2 = decoder_block(d1, e5, 512, padding="valid")
        d3 = decoder_block(d2, e4, 512)
        d4 = decoder_block(d3, e3, 512, dropout=False, padding="valid")
        d5 = decoder_block(d4, e2, 256, dropout=False)
        d6 = decoder_block(d5, e1, 128, dropout=False)
        conv = Conv2DTranspose(10, (4, 4), strides=(
            2, 2), padding='same', kernel_initializer=self.init)(d6)
        model = tf.keras.Model(inputs, conv)
        return model