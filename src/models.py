import tensorflow as tf

class ConvBlock(tf.keras.Model):
    def __init__(self, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=[3,3], padding="same")
        self.conv2 = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=[3,3], padding="same")
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.LeakyReLU()
        self.relu2 = tf.keras.layers.LeakyReLU()
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        return x

class BabyUnet(tf.keras.Model):
    def __init__(self):
        super(BabyUnet, self).__init__()
        self.total_loss = 0
        self.block1 = ConvBlock(16)
        self.block2 = ConvBlock(32)
        self.block3 = ConvBlock(32)
        self.block4 = ConvBlock(32)
        self.block5 = ConvBlock(16)
        self.block6 = ConvBlock(1)
        
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=2)
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2,2], strides=2)
        
        self.upsample1 = tf.keras.layers.UpSampling2D(interpolation='bilinear')
        self.upsample2 = tf.keras.layers.UpSampling2D(interpolation='bilinear')
    
    def call(self, inputs):
        x = self.block1(inputs)
        x1 = x
        x = self.pool1(x)
        x = self.block2(x)
        x2 = x
        x = self.pool2(x)
        x = self.block3(x)
        
        x = self.upsample1(x)
        x = tf.concat([x, x2], axis=-1)
        x = self.block4(x)
        x = self.upsample2(x)
        x1 = tf.concat([x, x1], axis=-1)
        x = self.block5(x)
        x = self.block6(x)
        return x