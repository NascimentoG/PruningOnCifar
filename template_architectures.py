import tensorflow as tf
from keras.layers import Conv2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense, Input, DepthwiseConv2D, ZeroPadding2D
from keras.regularizers import l2
from keras.models import Model
import keras.backend as K

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 name=''):
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),
                  name='Conv2D_{}'.format(name))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(name='BatchNorm1_{}'.format(name))(x)
        if activation is not None:
            x = Activation(activation, name='Act1_{}'.format(name))(x)
    else:
        if batch_normalization:
            x = BatchNormalization(name='BatchNorm2_{}'.format(name))(x)
        if activation is not None:
            x = Activation(activation, name='Act2_{}'.format(name))(x)
        x = conv(x)
    return x

def ResNet(input_shape, depth_block, filters=[], iter=0, num_classes=10):
    num_filters = 16
    i = 0
    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, num_filters=filters.pop(0))
    i = i + 1
    for stack in range(3):
        num_res_blocks = depth_block[stack]
        for res_block in range(num_res_blocks):
            layer_name = str(stack) + '_' + str(res_block) + '_' + str(iter)
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(inputs=x,
                             num_filters=filters.pop(0),
                             strides=strides,
                             name=layer_name + '_1')
            i = i + 1
            y = resnet_layer(inputs=y,
                             num_filters=filters.pop(0),
                             activation=None,
                             name=layer_name + '_2')
            i = i + 1
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x,
                                 num_filters=filters.pop(0),
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 name=layer_name + '_3')
                i = i + 1
            x = Add()([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    model = Model(inputs=inputs, outputs=outputs)
    return model

def relu6(x):
    return K.relu(x, max_value=6)

def _obtain_input_shape(input_shape,
                        default_size,
                        min_size,
                        data_format,
                        include_top):
    if data_format == 'channels_first':
        default_shape = (3, default_size, default_size)
    else:
        default_shape = (default_size, default_size, 3)
    if include_top:
        if input_shape is not None:
            if input_shape != default_shape:
                raise ValueError('When setting`include_top=True`, '
                                 '`input_shape` should be ' + str(default_shape) + '.')
        input_shape = default_shape
    else:
        if data_format == 'channels_first':
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError('`input_shape` must be a tuple of three integers.')
                if input_shape[0] != 3:
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[1] is not None and input_shape[1] < min_size) or
                   (input_shape[2] is not None and input_shape[2] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) + ', got '
                                     '`input_shape=' + str(input_shape) + '`')
            else:
                input_shape = (3, None, None)
        else:
            if input_shape is not None:
                if len(input_shape) != 3:
                    raise ValueError('`input_shape` must be a tuple of three integers.')
                if input_shape[-1] != 3:
                    raise ValueError('The input must have 3 channels; got '
                                     '`input_shape=' + str(input_shape) + '`')
                if ((input_shape[0] is not None and input_shape[0] < min_size) or
                   (input_shape[1] is not None and input_shape[1] < min_size)):
                    raise ValueError('Input size must be at least ' +
                                     str(min_size) + 'x' + str(min_size) + ', got '
                                     '`input_shape=' + str(input_shape) + '`')
            else:
                input_shape = (None, None, 3)
    return input_shape

def correct_pad(backend, inputs, kernel_size):
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    in_channels = backend.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = filters[0]
    pointwise_filters = filters[1]
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        x = Conv2D(filters[0],
                  kernel_size=1,
                  padding='same',
                  use_bias=False,
                  activation=None,
                  name=prefix + 'expand')(x)
        x = BatchNormalization(axis=channel_axis,
                              epsilon=1e-3,
                              momentum=0.999,
                              name=prefix + 'expand_BN')(x)
        x = Activation(relu6, name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    if stride == 2:
        x = ZeroPadding2D(padding=correct_pad(backend, x, 3),
                         name=prefix + 'pad')(x)
    x = DepthwiseConv2D(kernel_size=3,
                       strides=stride,
                       activation=None,
                       use_bias=False,
                       padding='same' if stride == 1 else 'valid',
                       name=prefix + 'depthwise')(x)
    x = BatchNormalization(axis=channel_axis,
                          epsilon=1e-3,
                          momentum=0.999,
                          name=prefix + 'depthwise_BN')(x)
    x = Activation(relu6, name=prefix + 'depthwise_relu')(x)
    x = Conv2D(filters[1],
              kernel_size=1,
              padding='same',
              use_bias=False,
              activation=None,
              name=prefix + 'project')(x)
    x = BatchNormalization(axis=channel_axis,
                          epsilon=1e-3,
                          momentum=0.999,
                          name=prefix + 'project_BN')(x)

    if in_channels == pointwise_filters and stride == 1:
        return Add(name=prefix + 'add')([inputs, x])
    return x

def MobileNetV2(input_shape=None,
                alpha=1.0,
                include_top=True,
                weights=None,
                input_tensor=None,
                pooling=None,
                num_classes=1000,
                initial_reduction=False,
                depth_block=[2, 3, 4, 3, 3],
                filters=[],
                **kwargs):
    if backend.image_data_format() == 'channels_first':
        rows = input_shape[1]
        cols = input_shape[2]
    else:
        rows = input_shape[0]
        cols = input_shape[1]

    if rows == cols and rows in [96, 128, 160, 192, 224]:
        default_size = rows
    else:
        default_size = 224

    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    img_input = Input(shape=input_shape)

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    if initial_reduction:
        first_block_filters = filters.pop(0)
        x = ZeroPadding2D(padding=correct_pad(backend, img_input, 3),
                         name='Conv1_pad')(img_input)
        x = Conv2D(first_block_filters,
                  kernel_size=3,
                  strides=(2, 2),
                  padding='valid',
                  use_bias=False,
                  name='Conv1')(x)
        x = BatchNormalization(axis=channel_axis,
                              epsilon=1e-3,
                              momentum=0.999,
                              name='bn_Conv1')(x)
        x = Activation(relu6, name='Conv1_relu')(x)
    else:
        x = img_input

    filters_block = [24, 32, 64, 96, 160]
    x = _inverted_res_block(x, filters=[3, filters.pop(0)], alpha=alpha, stride=1,
                            expansion=1, block_id=0)

    block_id = 1
    for stage in range(len(depth_block)):
        num_blocks = depth_block[stage]
        n_filters = filters_block[stage]

        for block in range(num_blocks):
            if block == 0 and n_filters != 96:
                x = _inverted_res_block(x, filters=[filters.pop(0), filters.pop(0)],
                                        alpha=alpha, stride=2, expansion=6, block_id=block_id)
            else:
                x = _inverted_res_block(x, filters=[filters.pop(0), filters.pop(0)],
                                        alpha=alpha, stride=1, expansion=6, block_id=block_id)
            block_id += 1

    x = _inverted_res_block(x, filters=[filters.pop(0), filters.pop(0)],
                            alpha=alpha, stride=1, expansion=6, block_id=block_id)

    x = Conv2D(filters.pop(0),
               kernel_size=1,
               use_bias=False,
               name='Conv_1')(x)
    x = BatchNormalization(axis=channel_axis,
                           epsilon=1e-3,
                           momentum=0.999,
                           name='Conv_1_bn')(x)
    x = Activation(relu6, name='out_relu')(x)

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = Dense(num_classes, activation='softmax', name='Logits')(x)
    else:
        if pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)

    model = Model(img_input, x, name=f'mobilenetv2_{alpha}_{rows}')
    return model


# Transformer
class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({'patch_size': self.patch_size})
        return config

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = tf.keras.layers.Dense(units=projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches,
            output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config()
        config.update({'num_patches': self.num_patches, 'projection_dim': self.projection_dim})
        return config

def Transformer(input_shape, projection_dim, num_heads, n_classes):
    inputs = tf.keras.Input(shape=input_shape)
    patches = Patches(4)(inputs)
    encoded_patches = PatchEncoder((input_shape[0] // 4) ** 2, projection_dim)(patches)

    for i in range(len(num_heads)):
        x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads[i], key_dim=projection_dim)(x1, x1)
        x2 = Add()([attention_output, encoded_patches])
        x3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = Dense(projection_dim, activation=tf.nn.gelu)(x3)
        encoded_patches = Add()([x3, x2])

    encoded_patches = Flatten()(encoded_patches)
    if n_classes == 2:
        outputs = Dense(n_classes, activation='sigmoid')(encoded_patches)
    else:
        outputs = Dense(n_classes, activation='softmax')(encoded_patches)

    model = Model(inputs, outputs)
    return model

 
