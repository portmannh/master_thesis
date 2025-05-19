"""
Model inspired by the U-Sleep model from: Perslev et al. 2021
Outputs a class label for every timepoint
The model outputs logits and not probabilities, so a softmax layer is needed to get probabilities
"""

import tensorflow as tf
import os

# set random seed for reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
tf.random.set_seed(seed)

from tensorflow.keras import layers


# encoder block
class Encoder(tf.keras.layers.Layer):
    def __init__(self, filters, activation='elu', padding='same', name=None):
        super(Encoder, self).__init__(name=name)

        # save parameters as instance attributes
        self.filters = filters
        self.activation_str = activation
        self.padding = padding

        # convolution and activation
        self.conv = layers.Conv1D(
            filters=filters, kernel_size=9, strides=1, padding=padding
            )
        
        # activation function
        self.activation = layers.Activation(activation)

        # batch normalization (of features/channels)
        self.norm = layers.BatchNormalization(axis=-1)

        # max pooling
        self.pool = layers.MaxPool1D(pool_size=2, strides=2, data_format='channels_last')


    def call(self, inputs, training=False):
        # convolution, activation, and batch normalization
        x = self.conv(inputs)
        x = self.activation(x)
        x = self.norm(x, training=training)
        
        # apply zero padding if needed (if odd number of timepoints)
        def if_odd():
            return layers.ZeroPadding1D(padding=(0,1))(x)
        def if_even():
            return x
        
        # check if input length is odd or even
        # apply zero padding if needed (if odd number of timepoints)
        x = tf.cond(tf.shape(x)[1] % 2 != 0, if_odd, if_even)

        # residual connection
        res = x
 
        # max pooling
        x = self.pool(x)

        return x, res
    
    def get_config(self):
        config = super(Encoder, self).get_config()
        config.update({
            'filters': self.filters,
            'activation': self.activation_str,
            'padding': self.padding
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# decoder block
class Decoder(tf.keras.layers.Layer):
    def __init__(self, filters, activation='elu', padding='same', name=None):
        super(Decoder, self).__init__(name=name)

        # save parameters as instance attributes
        self.filters = filters
        self.activation_str = activation
        self.padding = padding

        # nearest-neighbor upsampling (repeats each temporal step size times along the time axis)
        self.upsample = layers.UpSampling1D(size=2)

        # convolution
        self.conv1 = layers.Conv1D(
            filters=filters, kernel_size=9, strides=1, padding=padding
        )

        self.conv2 = layers.Conv1D(
            filters=filters, kernel_size=9, strides=1, padding=padding
        )

        # activation function
        self.activation = layers.Activation(activation)

        # batch normalization
        self.norm1 = layers.BatchNormalization(axis=-1)
        self.norm2 = layers.BatchNormalization(axis=-1)


    # cropping function to match length of residual connection
    def crop_to_match(self, x, residual):
        # if x's length is greater than residuals's length, crop x to match
        def crop_x():
            crop_length = tf.shape(x)[1] - tf.shape(residual)[1]
            return x[:, :tf.shape(x)[1] - crop_length, :]
        def no_crop():
            return x
        # check if x's length is greater than residual's length
        x = tf.cond(tf.shape(x)[1] > tf.shape(residual)[1], crop_x, no_crop)
        return x
 
    def call(self, inputs, residual, training=False):
        # upsampling, convolution, activation, and batchnorm
        x = self.upsample(inputs)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.norm1(x, training=training)

        # apply cropping if needed
        x = self.crop_to_match(x, residual)
        

        # concatenate with residual connection along feature axis
        x = layers.Concatenate(axis=-1)([x, residual])

        # convolution, activation, and batchnorm
        x = self.conv2(x)
        x = self.activation(x)
        x = self.norm2(x, training=training)

        return x
    
    # Add get_config method to save the configuration of the layer
    def get_config(self):
        config = super(Decoder, self).get_config()
        config.update({
            'filters': self.filters,
            'activation': self.activation_str,
            'padding': self.padding
        })
        return config

    # Add from_config method to reload the layer
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

# bottleneck block indluding convolution + batch norm
class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filters, activation='elu', padding='same', name=None):
        super(BottleNeck, self).__init__(name=name)

        # save parameters as instance attributes
        self.filters = filters
        self.activation_str = activation
        self.padding = padding

        # convolution
        self.conv = layers.Conv1D(filters=filters, kernel_size=9, strides=1, padding=padding, activation=activation)
        
        # batch normaliztion
        self.norm = layers.BatchNormalization(axis=-1)

    
    def call(self, inputs, training=False):
        x = self.conv(inputs)
        x = self.norm(x, training=training)

        return x
    
    # Add get_config method to save the configuration of the layer
    def get_config(self):
        config = super(BottleNeck, self).get_config()
        config.update({
            'filters': self.filters,
            'activation': self.activation_str,
            'padding': self.padding
        })
        return config

    # Add from_config method to reload the layer
    @classmethod
    def from_config(cls, config):
        return cls(**config)


# whole model
class MyUSleep(tf.keras.Model):
    def __init__(self, num_classes=5, in_shape=(None, 4)):
        super(MyUSleep, self).__init__()

        # encoding
        self.encode1 = Encoder(filters=6, name='encoder1')
        self.encode2 = Encoder(filters=9, name='encoder2')
        self.encode3 = Encoder(filters=11, name='encoder3')
        self.encode4 = Encoder(filters=15, name='encoder4')
        self.encode5 = Encoder(filters=20, name='encoder5')
        self.encode6 = Encoder(filters=28, name='encoder6')
        self.encode7 = Encoder(filters=40, name='encoder7')
        self.encode8 = Encoder(filters=55, name='encoder8')
        self.encode9 = Encoder(filters=77, name='encoder9')
        self.encode10 = Encoder(filters=108, name='encoder10')
        self.encode11 = Encoder(filters=152, name='encoder11')
        self.encode12 = Encoder(filters=214, name='encoder12')

        # convolution + batch norm
        self.conv_norm = BottleNeck(filters=306, name='bottleneck')

        # decoding
        self.decode12 = Decoder(filters=214, name='decoder12')
        self.decode11 = Decoder(filters=152, name='decoder11')
        self.decode10 = Decoder(filters=108, name='decoder10')
        self.decode9 = Decoder(filters=77, name='decoder9')
        self.decode8 = Decoder(filters=55, name='decoder8')
        self.decode7 = Decoder(filters=40, name='decoder7')
        self.decode6 = Decoder(filters=28, name='decoder6')
        self.decode5 = Decoder(filters=20, name='decoder5')
        self.decode4 = Decoder(filters=15, name='decoder4')
        self.decode3 = Decoder(filters=11, name='decoder3')
        self.decode2 = Decoder(filters=9, name='decoder2')
        self.decode1 = Decoder(filters=6, name='decoder1')

        # classifier 
        self.conv1 = layers.Conv1D(
            filters=6, kernel_size=1, strides=1, padding='same', activation='tanh', name='classifier_conv1'
            )
        
        # if we want timepoints as output we don't need pooling layer
        
        self.conv2 = layers.Conv1D(
            filters=num_classes, kernel_size=1, strides=1, padding='same', activation='elu', name='classifier_conv2'
            )
        
        self.conv3 = layers.Conv1D(
            filters=num_classes, kernel_size=1, strides=1, padding='same', name='classifier_conv3'
        )

    # cropping function to match output length to input length
    # this is important if the input length is uneven, otherwise the output will be one timepoint longer than the input
    def crop_to_match_input(self, x, input):
        # Calculate crop length dynamically
        crop_length = tf.shape(x)[1] - tf.shape(input)[1]

        # Check if cropping is needed
        def crop_fn():
            return x[:, :tf.shape(input)[1], :]  # Crop to match input length

        def no_crop_fn():
            return x  # No cropping needed

        # Use tf.cond to decide whether to crop
        x = tf.cond(crop_length > 0, crop_fn, no_crop_fn)
        return x

    
    def call(self, x, training=False):
        # save this to keep original dimensions
        original_input = x

        # encoding
        x, res1 = self.encode1(x)
        x, res2 = self.encode2(x)
        x, res3 = self.encode3(x)
        x, res4 = self.encode4(x)
        x, res5 = self.encode5(x)
        x, res6 = self.encode6(x)
        x, res7 = self.encode7(x)
        x, res8 = self.encode8(x)
        x, res9 = self.encode9(x)
        x, res10 = self.encode10(x)
        x, res11 = self.encode11(x)
        x, res12 = self.encode12(x)

        # bottleneck
        x = self.conv_norm(x)
        features = x

        # decoding
        x = self.decode12(x, res12)
        x = self.decode11(x, res11)
        x = self.decode10(x, res10)
        x = self.decode9(x, res9)
        x = self.decode8(x, res8)
        x = self.decode7(x, res7)
        x = self.decode6(x, res6)
        x = self.decode5(x, res5)
        x = self.decode4(x, res4)
        x = self.decode3(x, res3)
        x = self.decode2(x, res2)
        x = self.decode1(x, res1)

        # crop if input length was uneven
        x = self.crop_to_match_input(x, original_input)

        # classifier
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # when training only return logits (x), when evaluating return logits and features from bottleneck layer
        if not training:
            return x, features
        else:
            return x