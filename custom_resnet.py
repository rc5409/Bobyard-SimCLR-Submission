import tensorflow as tf

BATCH_NORM_EPSILON = 1e-5

class BatchNormRelu(tf.keras.layers.Layer):
    def __init__(self, relu=True, init_zero=False, center=True, scale=True,
                 data_format='channels_last', **kwargs):
        super(BatchNormRelu, self).__init__(**kwargs)
        self.relu = relu
        gamma_initializer = tf.zeros_initializer() if init_zero else tf.ones_initializer()
        axis = 1 if data_format == 'channels_first' else -1
        
        self.bn = tf.keras.layers.BatchNormalization(
            axis=axis,
            momentum=0.9,
            epsilon=BATCH_NORM_EPSILON,
            center=center,
            scale=scale,
            gamma_initializer=gamma_initializer
        )

    def call(self, inputs, training=False):
        x = self.bn(inputs, training=training)
        if self.relu:
            x = tf.nn.relu(x)
        return x

class FixedPadding(tf.keras.layers.Layer):
    def __init__(self, kernel_size, data_format='channels_last', **kwargs):
        super(FixedPadding, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.data_format = data_format

    def call(self, inputs, training=False):
        pad_total = self.kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        
        if self.data_format == 'channels_first':
            padded = tf.pad(
                inputs,
                [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]]
            )
        else:
            padded = tf.pad(
                inputs,
                [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]
            )
        return padded

class Conv2dFixedPadding(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, data_format='channels_last', **kwargs):
        super(Conv2dFixedPadding, self).__init__(**kwargs)
        self.fixed_padding = FixedPadding(kernel_size, data_format) if strides > 1 else None
        self.conv2d = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=('SAME' if strides == 1 else 'VALID'),
            use_bias=False,
            kernel_initializer=tf.keras.initializers.VarianceScaling(),
            data_format=data_format
        )

    def call(self, inputs, training=False):
        if self.fixed_padding is not None:
            inputs = self.fixed_padding(inputs, training=training)
        return self.conv2d(inputs)

class SE_Layer(tf.keras.layers.Layer):
    """
    Squeeze-and-Excitation layer that automatically infers # of channels from input_shape.
    """
    def __init__(self, se_ratio=0.25, data_format='channels_last', **kwargs):
        super(SE_Layer, self).__init__(**kwargs)
        self.se_ratio = se_ratio
        self.data_format = data_format

    def build(self, input_shape):
        self.filters = input_shape[-1]
        reduced_filters = max(1, int(self.filters * self.se_ratio))

        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(
            data_format=self.data_format
        )
        self.fc1 = tf.keras.layers.Dense(reduced_filters, activation='relu', use_bias=False)
        self.fc2 = tf.keras.layers.Dense(self.filters, activation='sigmoid', use_bias=False)
        super(SE_Layer, self).build(input_shape)

    def call(self, inputs):
        se_tensor = self.global_avg_pool(inputs)  
        se_tensor = self.fc1(se_tensor)           
        se_tensor = self.fc2(se_tensor)           
        if self.data_format == 'channels_last':
            se_tensor = tf.reshape(se_tensor, [-1, 1, 1, self.filters])
        else:
            se_tensor = tf.reshape(se_tensor, [-1, self.filters, 1, 1])

        return inputs * se_tensor

class BottleneckBlock(tf.keras.layers.Layer):
    def __init__(self, filters, strides, use_projection=False,
                 data_format='channels_last', **kwargs):
        super(BottleneckBlock, self).__init__(**kwargs)
        self.projection_layers = []
        if use_projection:
            filters_out = 4 * filters
            self.projection_layers.append(
                Conv2dFixedPadding(filters_out, kernel_size=1, strides=strides,
                                   data_format=data_format)
            )
            self.projection_layers.append(
                BatchNormRelu(relu=False, data_format=data_format)
            )
        
        self.conv_relu_layers = [
            Conv2dFixedPadding(filters, kernel_size=1, strides=1, data_format=data_format),
            BatchNormRelu(data_format=data_format),

            Conv2dFixedPadding(filters, kernel_size=3, strides=strides, data_format=data_format),
            BatchNormRelu(data_format=data_format),

            Conv2dFixedPadding(4 * filters, kernel_size=1, strides=1, data_format=data_format),
            BatchNormRelu(relu=False, data_format=data_format)
        ]

        self.se_layer = SE_Layer(se_ratio=0.25, data_format=data_format)

    def call(self, inputs, training=False):
        shortcut = inputs
        for layer in self.projection_layers:
            shortcut = layer(shortcut, training=training)

        x = inputs
        for layer in self.conv_relu_layers:
            x = layer(x, training=training)


        x = self.se_layer(x)

        return tf.nn.relu(x + shortcut)

class BlockGroup(tf.keras.layers.Layer):
    def __init__(self, filters, block_fn, blocks, strides,
                 data_format='channels_last', **kwargs):
        super(BlockGroup, self).__init__(**kwargs)
        self.layers = []
        self.layers.append(block_fn(filters, strides, use_projection=True, data_format=data_format))

        for _ in range(1, blocks):
            self.layers.append(block_fn(filters, 1, data_format=data_format))

    def call(self, inputs, training=False):
        for layer in self.layers:
            inputs = layer(inputs, training=training)
        return inputs

class ResNet(tf.keras.Model):
    def __init__(self, resnet_depth, width_multiplier=1, data_format='channels_last', **kwargs):
        super(ResNet, self).__init__(**kwargs)
        self.data_format = data_format

        model_params = {
            50: {'block': BottleneckBlock, 'layers': [3, 4, 6, 3]},

        }
        params = model_params[resnet_depth]
        block_fn = params['block']
        layers = params['layers']

        self.initial_conv = Conv2dFixedPadding(64 * width_multiplier, kernel_size=7, strides=2,
                                               data_format=data_format)
        self.initial_bn_relu = BatchNormRelu(data_format=data_format)
        self.initial_max_pool = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2,
                                                              padding='SAME',
                                                              data_format=data_format)

        self.block_groups = [
            BlockGroup(64 * width_multiplier,  block_fn, layers[0], strides=1,
                       data_format=data_format),
            BlockGroup(128 * width_multiplier, block_fn, layers[1], strides=2,
                       data_format=data_format),
            BlockGroup(256 * width_multiplier, block_fn, layers[2], strides=2,
                       data_format=data_format),
            BlockGroup(512 * width_multiplier, block_fn, layers[3], strides=2,
                       data_format=data_format),
        ]

    def call(self, inputs, training=False):
        x = self.initial_conv(inputs, training=training)
        x = self.initial_bn_relu(x, training=training)
        x = self.initial_max_pool(x, training=training)

        for block_group in self.block_groups:
            x = block_group(x, training=training)

        if self.data_format == 'channels_last':
            x = tf.reduce_mean(x, [1, 2])
        else:
            x = tf.reduce_mean(x, [2, 3])
        return x

def build_resnet50(width_multiplier=1, data_format='channels_last'):
    return ResNet(resnet_depth=50, width_multiplier=width_multiplier,
                  data_format=data_format)
