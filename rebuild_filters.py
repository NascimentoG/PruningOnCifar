import numpy as np
import random
import copy
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import rebuild_layers as rl  # It implements some particular functions we need to use here

isFiltersAvailable = True

# Used by MobileNet
def relu6(x):
    return K.relu(x, max_value=6)


def rw_bn(w, index):
    w[0] = np.delete(w[0], index)
    w[1] = np.delete(w[1], index)
    w[2] = np.delete(w[2], index)
    w[3] = np.delete(w[3], index)
    return w


# --- Helpers para criação/atribuição de pesos (compatível TF 2.19) --- #
def _build_and_set_weights(layer, weights):
    """
    Build layer using an inferred input shape (from weights) and set weights.
    This avoids using the deprecated `weights=` constructor argument.
    """
    try:
        # Conv kernels: (kh, kw, in_channels, out_channels)
        if isinstance(layer, Conv2D):
            in_ch = weights[0].shape[2]
            layer.build((None, None, None, int(in_ch)))
        # Depthwise: (kh, kw, in_channels, depth_multiplier)
        elif isinstance(layer, DepthwiseConv2D):
            in_ch = weights[0].shape[2]
            layer.build((None, None, None, int(in_ch)))
        # Dense: weights[0] shape (in_units, out_units)
        elif isinstance(layer, Dense):
            in_units = weights[0].shape[0]
            layer.build((None, int(in_units)))
        # BatchNormalization: gamma shape (channels,)
        elif isinstance(layer, BatchNormalization):
            channels = weights[0].shape[0]
            # build with shape (None, channels). BN supports several ranks; this is enough to create variables.
            layer.build((None, int(channels)))
        else:
            # Fallback: try to build with generic vector shape if available
            if len(weights) > 0 and hasattr(weights[0], 'shape'):
                shp = tuple(weights[0].shape)
                layer.build((None,) + shp)
    except Exception:
        # If build fails, try calling once with a minimal input when used in the model graph (deferred)
        pass

    # Now set weights
    if len(weights) > 0:
        layer.set_weights(weights)
    return layer


def create_Conv2D_from_conf(config, weights):
    n_filters = int(weights[0].shape[-1])
    layer = Conv2D(
        filters=n_filters,
        kernel_size=tuple(config['kernel_size']) if isinstance(config.get('kernel_size'), (list, tuple, np.ndarray)) else config.get('kernel_size'),
        strides=tuple(config['strides']) if isinstance(config.get('strides'), (list, tuple, np.ndarray)) else config.get('strides'),
        padding=config.get('padding'),
        data_format=config.get('data_format'),
        dilation_rate=tuple(config['dilation_rate']) if isinstance(config.get('dilation_rate'), (list, tuple, np.ndarray)) else config.get('dilation_rate'),
        activation=config.get('activation'),
        use_bias=config.get('use_bias'),
        name=config.get('name'),
        trainable=config.get('trainable'),
        kernel_regularizer=config.get('kernel_regularizer'),
        bias_regularizer=config.get('bias_regularizer'),
        activity_regularizer=config.get('activity_regularizer'),
        kernel_constraint=config.get('kernel_constraint'),
        bias_constraint=config.get('bias_constraint'),
    )
    return _build_and_set_weights(layer, weights)


def create_depthwise_from_config(config, weights):
    layer = DepthwiseConv2D(
        kernel_size=tuple(config['kernel_size']) if isinstance(config.get('kernel_size'), (list, tuple, np.ndarray)) else config.get('kernel_size'),
        strides=tuple(config['strides']) if isinstance(config.get('strides'), (list, tuple, np.ndarray)) else config.get('strides'),
        padding=config.get('padding'),
        data_format=config.get('data_format'),
        dilation_rate=tuple(config['dilation_rate']) if isinstance(config.get('dilation_rate'), (list, tuple, np.ndarray)) else config.get('dilation_rate'),
        depth_multiplier=config.get('depth_multiplier'),
        activation=config.get('activation'),
        use_bias=config.get('use_bias'),
        name=config.get('name'),
        trainable=config.get('trainable'),
        depthwise_regularizer=config.get('depthwise_regularizer'),
        depthwise_constraint=config.get('depthwise_constraint'),
        depthwise_initializer=config.get('depthwise_initializer'),
        activity_regularizer=config.get('activity_regularizer'),
        bias_regularizer=config.get('bias_regularizer'),
        bias_constraint=config.get('bias_constraint'),
    )
    return _build_and_set_weights(layer, weights)


def create_Dense_from_conf(config, weights):
    layer = Dense(
        units=int(config.get('units')),
        activation=config.get('activation'),
        use_bias=config.get('use_bias'),
        name=config.get('name'),
        trainable=config.get('trainable'),
        kernel_regularizer=config.get('kernel_regularizer'),
        bias_regularizer=config.get('bias_regularizer'),
        activity_regularizer=config.get('activity_regularizer'),
        kernel_constraint=config.get('kernel_constraint'),
        bias_constraint=config.get('bias_constraint'),
    )
    return _build_and_set_weights(layer, weights)


def create_BN_from_weights(name, weights, epsilon=None, momentum=None):
    # weights expected in order: [gamma, beta, moving_mean, moving_variance]
    cfg = {'name': name}
    layer = BatchNormalization(name=name,
                               epsilon=epsilon if epsilon is not None else 0.001,
                               momentum=momentum if momentum is not None else 0.99)
    return _build_and_set_weights(layer, weights)


# --- Funções de remoção/ajuste de pesos (mantive sua lógica) --- #
def rw_cn(index_model, idx_pruned, model):
    # This function removes the weights of the Conv2D considering the previous pruning in other Conv2D
    config = model.get_layer(index=index_model).get_config()
    weights = model.get_layer(index=index_model).get_weights()
    weights[0] = np.delete(weights[0], idx_pruned, axis=2)
    return create_Conv2D_from_conf(config, weights)


def remove_conv_weights(index_model, idxs, model):
    config, weights = (model.get_layer(index=index_model).get_config(),
                       model.get_layer(index=index_model).get_weights())
    weights[0] = np.delete(weights[0], idxs, axis=3)
    # if bias exists
    if len(weights) > 1 and weights[1] is not None:
        weights[1] = np.delete(weights[1], idxs)
        config['filters'] = int(weights[1].shape[0])
    else:
        config['filters'] = int(weights[0].shape[-1])
    return idxs, config, weights


def remove_convMobile_weights(index_model, idxs, model):
    config, weights = (model.get_layer(index=index_model).get_config(),
                       model.get_layer(index=index_model).get_weights())
    weights[0] = np.delete(weights[0], idxs, axis=3)
    config['filters'] = int(weights[0].shape[-1])
    return idxs, config, weights


# --- Rebuild ResNet (CIFAR style) --- #
def rebuild_resnet(model, blocks, layer_filters, num_classes=10):
    num_filters = 16
    num_res_blocks = blocks

    inputs = Input(shape=(model.inputs[0].shape[1],
                          model.inputs[0].shape[2],
                          model.inputs[0].shape[3]))

    # The first block is not allowed to prune
    _, config, weights = remove_conv_weights(1, [], model)
    conv = create_Conv2D_from_conf(config, weights)

    H = conv(inputs)

    # BatchNormalization (instantiate and set weights)
    bn_w = model.get_layer(index=2).get_weights()
    bn_layer = create_BN_from_weights(name=model.get_layer(index=2).name, weights=bn_w)
    H = bn_layer(H)

    H = Activation.from_config(model.get_layer(index=3).get_config())(H)
    x = H

    i = 4

    remove_Conv2D = [item[1] for item in layer_filters]
    remove_Conv2D.reverse()
    layer_block = False
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks[stack]):

            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample

            # This is the layer we can prune
            idx_previous, config, weights = remove_conv_weights(i, remove_Conv2D.pop(), model)
            conv = create_Conv2D_from_conf(config, weights)
            i = i + 1
            y = conv(x)
            wb = model.get_layer(index=i).get_weights()
            y = create_BN_from_weights(name=model.get_layer(index=i).name, weights=rw_bn(wb, idx_previous))(y)
            i = i + 1
            y = Activation.from_config(model.get_layer(index=i).get_config())(y)
            i = i + 1

            # Second Module
            conv = rw_cn(index_model=i, idx_pruned=idx_previous, model=model)
            i = i + 1
            y = conv(y)
            # Here there's logic depending on block ordering
            if layer_block is False:
                bn_w2 = model.get_layer(index=i).get_weights()
                y = create_BN_from_weights(name=model.get_layer(index=i).name, weights=bn_w2)(y)
            else:
                bn_w2 = model.get_layer(index=i + 1).get_weights()
                y = create_BN_from_weights(name=model.get_layer(index=i + 1).name, weights=bn_w2)(y)
                layer_block = False
            i = i + 1

            if stack > 0 and res_block == 0:
                # linear projection residual shortcut connection to match changed dims
                _, config, weights = remove_conv_weights(i - 1, [], model)
                conv = create_Conv2D_from_conf(config, weights)
                x = conv(x)
                i = i + 1

            x = Add()([x, y])
            i = i + 1
            x = Activation.from_config(model.get_layer(index=i).get_config())(x)
            i = i + 1
        num_filters *= 2
        layer_block = True

    # Add classifier on top.
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)

    layer = model.get_layer(index=-1)
    config = layer.get_config()
    weights = layer.get_weights()
    dense_layer = create_Dense_from_conf(config, weights)
    outputs = dense_layer(y)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# --- Rebuild ResNet (ImageNet - with BN naming) --- #
def rebuild_resnetBN(model, blocks, layer_filters, iter=0, num_classes=1000):
    stacks = len(blocks)
    num_filters = 64
    layer_filters = dict(layer_filters)

    inputs = Input(shape=(model.inputs[0].shape[1],
                          model.inputs[0].shape[2],
                          model.inputs[0].shape[3]))

    # ZeroPadding2D(padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = ZeroPadding2D.from_config(config=model.get_layer(index=1).get_config())(inputs)

    # Conv2D(64, 7, strides=2, use_bias=True, name='conv1_conv')(x)
    _, config, weights = remove_conv_weights(2, [], model)
    conv = create_Conv2D_from_conf(config, weights)
    x = conv(x)

    # BatchNormalization
    bn3_w = model.get_layer(index=3).get_weights()
    x = create_BN_from_weights(name='BN00' + str(iter), weights=bn3_w, epsilon=1.001e-5)(x)

    # Activation
    x = Activation.from_config(config=model.get_layer(index=4).get_config())(x)

    # ZeroPadding2D and MaxPool
    x = ZeroPadding2D.from_config(config=model.get_layer(index=5).get_config())(x)
    x = MaxPooling2D.from_config(config=model.get_layer(index=6).get_config())(x)

    i = 7
    for stage in range(0, stacks):
        num_res_blocks = blocks[stage]

        # First Layer Block
        shortcut = x

        _, config, weights = remove_conv_weights(i, [], model)
        conv = create_Conv2D_from_conf(config, weights)
        x = conv(x)
        i = i + 1

        bn_w = model.get_layer(index=i).get_weights()
        x = create_BN_from_weights(name='BN' + str(i) + str(iter), weights=bn_w, epsilon=1.001e-5)(x)
        i = i + 1

        x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
        i = i + 1

        _, config, weights = remove_conv_weights(i, [], model)
        conv = create_Conv2D_from_conf(config, weights)
        x = conv(x)
        i = i + 1

        bn_w = model.get_layer(index=i).get_weights()
        x = create_BN_from_weights(name='BN' + str(i) + str(iter), weights=bn_w, epsilon=1.001e-5)(x)
        i = i + 1

        x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
        i = i + 1

        _, config, weights = remove_conv_weights(i + 1, [], model)
        conv = create_Conv2D_from_conf(config, weights)
        x = conv(x)

        bn_w_tmp = model.get_layer(index=i + 3).get_weights()
        x = create_BN_from_weights(name='BN' + str(i) + str(iter), weights=bn_w_tmp, epsilon=1.001e-5)(x)

        _, config, weights = remove_conv_weights(i, [], model)
        conv = create_Conv2D_from_conf(config, weights)
        shortcut = conv(shortcut)
        i = i + 2

        shortcut_bn_w = model.get_layer(index=i).get_weights()
        shortcut = create_BN_from_weights(name='BN' + str(i) + str(iter), weights=shortcut_bn_w, epsilon=1.001e-5)(shortcut)
        i = i + 1

        x = Add(name=model.get_layer(index=i).name)([shortcut, x])
        i = i + 2

        x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
        i = i + 1

        # end First Layer Block

        for res_block in range(2, num_res_blocks + 1):
            shortcut = x

            idx_previous, config, weights = remove_conv_weights(i, layer_filters.get(i, []), model)
            conv = create_Conv2D_from_conf(config, weights)
            x = conv(x)
            i = i + 1

            wb = model.get_layer(index=i).get_weights()
            x = create_BN_from_weights(name='BN' + str(i) + str(iter), weights=rw_bn(wb, idx_previous), epsilon=1.001e-5)(x)
            i = i + 1

            x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
            i = i + 1

            # next conv
            weights = model.get_layer(index=i).get_weights()
            config = model.get_layer(index=i).get_config()
            idxs = layer_filters.get(i, [])
            weights[0] = np.delete(weights[0], idxs, axis=3)
            if len(weights) > 1 and weights[1] is not None:
                weights[1] = np.delete(weights[1], idxs)
            # remove input channels affected by previous pruning
            weights[0] = np.delete(weights[0], idx_previous, axis=2)
            idx_previous = idxs
            conv = create_Conv2D_from_conf(config, weights)
            x = conv(x)
            i = i + 1

            wb = model.get_layer(index=i).get_weights()
            x = create_BN_from_weights(name='BN' + str(i) + str(iter), weights=rw_bn(wb, idx_previous), epsilon=1.001e-5)(x)
            i = i + 1

            x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
            i = i + 1

            weights = model.get_layer(index=i).get_weights()
            config = model.get_layer(index=i).get_config()
            weights[0] = np.delete(weights[0], idx_previous, axis=2)
            conv = create_Conv2D_from_conf(config, weights)
            x = conv(x)
            i = i + 1

            bn_w = model.get_layer(index=i).get_weights()
            x = create_BN_from_weights(name='BN' + str(i) + str(iter), weights=bn_w, epsilon=1.001e-5)(x)
            i = i + 1

            x = Add.from_config(config=model.get_layer(index=i).get_config())([shortcut, x])
            i = i + 1

            x = Activation.from_config(config=model.get_layer(index=i).get_config())(x)
            i = i + 1

        num_filters = num_filters * 2

    x = GlobalAveragePooling2D.from_config(config=model.get_layer(index=i).get_config())(x)
    i = i + 1

    weights = model.get_layer(index=i).get_weights()
    config = model.get_layer(index=i).get_config()
    dense_layer = create_Dense_from_conf(config, weights)
    x = dense_layer(x)

    model = Model(inputs, x, name='ResNetBN')
    return model


# --- Rebuild MobileNetV2 --- #
def rebuild_mobilenetV2(model, blocks, layer_filters, initial_reduction=False, num_classes=1000):
    blocks = np.append(blocks, 1)
    stacks = len(blocks)
    layer_filters = dict(layer_filters)

    inputs = Input(shape=(model.inputs[0].shape[1],
                          model.inputs[0].shape[2],
                          model.inputs[0].shape[3]))

    idx_previous = []
    i = 1
    if isinstance(model.get_layer(index=i), ZeroPadding2D):
        x = ZeroPadding2D.from_config(model.get_layer(index=i).get_config())(inputs)
        i = i + 1

        config, weights = model.get_layer(index=i).get_config(), model.get_layer(index=i).get_weights()
        x = create_Conv2D_from_conf(config, weights)(x)
        i = i + 1

        bn_w = model.get_layer(index=i).get_weights()
        x = create_BN_from_weights(name=model.get_layer(index=i).name, weights=bn_w, epsilon=1e-3, momentum=0.999)(x)
        i = i + 1

        x = Activation(relu6, name=model.get_layer(index=i).name)(x)
        i = i + 1

    else:
        x = inputs

    config, weights = model.get_layer(index=i).get_config(), model.get_layer(index=i).get_weights()
    x = create_depthwise_from_config(config, weights)(x)
    i = i + 1

    bn_w = model.get_layer(index=i).get_weights()
    x = create_BN_from_weights(name=model.get_layer(index=i).name, weights=bn_w, epsilon=1e-3, momentum=0.999)(x)
    i = i + 1

    x = Activation(relu6, name=model.get_layer(index=i).name)(x)
    i = i + 1

    idx_previous, config, weights = remove_convMobile_weights(i, layer_filters.get(i, []), model)
    x = create_Conv2D_from_conf(config, weights)(x)
    i = i + 1

    wb = model.get_layer(index=i).get_weights()
    x = create_BN_from_weights(name=model.get_layer(index=i).name, weights=rw_bn(wb, idx_previous), epsilon=1e-3, momentum=0.999)(x)
    i = i + 1

    id = 1
    for stage in range(0, stacks):
        num_blocks = blocks[stage]

        for mobile_block in range(0, num_blocks):
            prefix = 'block_{}_'.format(id)
            shortcut = x

            # 1x1 Convolution -- _expand
            idx = layer_filters.get(i, [])
            config, weights = model.get_layer(index=i).get_config(), model.get_layer(index=i).get_weights()
            weights[0] = np.delete(weights[0], idx, axis=3)

            # First block expand only
            if id == 1:
                weights[0] = np.delete(weights[0], idx_previous, axis=2)

            x = create_Conv2D_from_conf(config, weights)(x)
            i = i + 1

            wb = model.get_layer(index=i).get_weights()
            x = create_BN_from_weights(name=prefix + 'expand_BN', weights=rw_bn(wb, idx), epsilon=1e-3, momentum=0.999)(x)
            i = i + 1

            x = Activation(relu6, name=model.get_layer(index=i).name)(x)
            i = i + 1

            if isinstance(model.get_layer(index=i), ZeroPadding2D):  # stride==2
                x = ZeroPadding2D.from_config(model.get_layer(index=i).get_config())(x)
                i = i + 1

            # block_kth_depthwise
            config, weights = model.get_layer(index=i).get_config(), model.get_layer(index=i).get_weights()
            weights[0] = np.delete(weights[0], idx, axis=2)
            x = create_depthwise_from_config(config, weights)(x)
            i = i + 1

            wb = model.get_layer(index=i).get_weights()
            x = create_BN_from_weights(name=model.get_layer(index=i).name, weights=rw_bn(wb, idx), epsilon=1e-3, momentum=0.999)(x)
            i = i + 1

            x = Activation(relu6, name=model.get_layer(index=i).name)(x)
            i = i + 1

            # block_kth_project
            x = rw_cn(i, idx, model)(x)
            i = i + 1

            x = create_BN_from_weights(name=model.get_layer(index=i).name, weights=model.get_layer(index=i).get_weights(), epsilon=1e-3, momentum=0.999)(x)
            i = i + 1

            if isinstance(model.get_layer(index=i), Add):
                x = Add(name=model.get_layer(index=i).name)([shortcut, x])
                i = i + 1

            id = id + 1

    config, weights = model.get_layer(index=i).get_config(), model.get_layer(index=i).get_weights()
    x = create_Conv2D_from_conf(config, weights)(x)

    x = create_BN_from_weights(name=model.get_layer(index=i + 1).name, weights=model.get_layer(index=i + 1).get_weights(), epsilon=1e-3, momentum=0.999)(x)

    x = Activation(relu6, name=model.get_layer(index=i + 2).name)(x)

    x = GlobalAveragePooling2D.from_config(model.get_layer(index=i + 3).get_config())(x)

    config, weights = model.get_layer(index=i + 4).get_config(), model.get_layer(index=i + 4).get_weights()
    dense_layer = create_Dense_from_conf(config, weights)
    x = dense_layer(x)

    model = Model(inputs, x, name='MobileNetV2')
    return model


# --- Allowed layers discovery / helper utilities --- #
def allowed_layers_resnet(model):
    global isFiltersAvailable

    allowed_layers = []
    all_add = []
    n_filters = 0
    available_filters = 0

    for i in range(0, len(model.layers)):
        layer = model.get_layer(index=i)
        if isinstance(layer, Add):
            all_add.append(i)
        if isinstance(layer, Conv2D) and layer.strides == (2, 2) and layer.kernel_size != (1, 1):
            allowed_layers.append(i)

    if len(all_add) > 0:
        allowed_layers.append(all_add[0] - 5)

    for i in range(1, len(all_add)):
        allowed_layers.append(all_add[i] - 5)

    # To avoid bug due to keras architecture (i.e., order of layers)
    # This ensure that only Conv2D are "allowed layers"
    tmp = allowed_layers
    allowed_layers = []

    for i in tmp:
        if isinstance(model.get_layer(index=i), Conv2D):
            allowed_layers.append(i)
            layer = model.get_layer(index=i)
            config = layer.get_config()
            n_filters += int(config.get('filters', 0))

    available_filters = n_filters - len(allowed_layers)

    if available_filters == 0:
        isFiltersAvailable = False

    print(f"Numero de filtros nas camadas permitidas (PODA POR FILTRO) {available_filters} em {len(allowed_layers)}")
    return allowed_layers


def allowed_layers_resnetBN(model):
    allowed_layers = []
    all_add = []
    for i in range(0, len(model.layers)):
        if isinstance(model.get_layer(index=i), Add):
            all_add.append(i)

    for i in range(1, len(all_add) - 1):
        input_shape = model.get_layer(index=all_add[i]).output_shape
        output_shape = model.get_layer(index=all_add[i - 1]).output_shape
        # These are the valid blocks we can remove
        if input_shape == output_shape:
            allowed_layers.append(all_add[i] - 8)
            allowed_layers.append(all_add[i] - 5)

    return allowed_layers


def allowed_layers_mobilenetV2(model):
    allowed_layers = []
    for i in range(0, len(model.layers)):
        layer = model.get_layer(index=i)

        if isinstance(layer, Conv2D) and not isinstance(layer, DepthwiseConv2D):
            if 'expand' in layer.name:
                allowed_layers.append(i)

    return allowed_layers


def idx_to_conv2Didx(model, indices):
    # Convert index onto Conv2D index (required by pruning methods)
    idx_Conv2D = 0
    output = []
    for i in range(0, len(model.layers)):
        if isinstance(model.get_layer(index=i), Conv2D):
            if i in indices:
                output.append(idx_Conv2D)
            idx_Conv2D = idx_Conv2D + 1
    return output


def layer_to_prune_filters(model):
    # architecture_name is expected to be global or defined in scope where called
    if architecture_name.__contains__('ResNet'):
        if architecture_name.__contains__('50'):  # ImageNet architectures (ResNet50, 101 and 152)
            allowed_layers = allowed_layers_resnetBN(model)
        else:  # CIFAR-like architectures (low-resolution datasets)
            allowed_layers = allowed_layers_resnet(model)

    if architecture_name.__contains__('MobileNetV2'):
        allowed_layers = allowed_layers_mobilenetV2(model)

    # allowed_layers = idx_to_conv2Didx(model, allowed_layers)
    return allowed_layers


# --- Rebuild network entry point --- #
def rebuild_network(model, scores, p_filter, totalFiltersToRemove=0, wasPfilterZero=False):
    global isFiltersAvailable
    numberFiltersRemoved = 0
    scores = sorted(scores, key=lambda x: x[0])

    allowed_layers = [x[0] for x in scores]
    scores = [x[1] for x in scores]
    filtersToRemove = copy.deepcopy(scores)

    for i in range(0, len(scores)):
        num_remove = round(p_filter * len(scores[i]))
        numberFiltersRemoved += num_remove
        filtersToRemove[i] = np.argpartition(scores[i], num_remove)[:num_remove]

    layerSelectedList = [i for i in range(0, len(scores))]
    if totalFiltersToRemove != 0 and not wasPfilterZero:
        while ((totalFiltersToRemove - numberFiltersRemoved) != 0) and (len(layerSelectedList) != 0):
            layerSelected = random.choice(layerSelectedList)
            if (len(scores[layerSelected]) - (len(filtersToRemove[layerSelected])) - 1) > 0:
                filterToRemove = np.argpartition(scores[layerSelected],
                                                 (len(filtersToRemove[layerSelected]) + 1))[:(
                                                     len(filtersToRemove[layerSelected]) + 1)]
                filtersToRemove[layerSelected] = filterToRemove
                numberFiltersRemoved += 1
            else:
                layerSelectedList.remove(layerSelected)

    if len(layerSelectedList) == 0:
        isFiltersAvailable = False
        print(f"Faltam remover {totalFiltersToRemove - numberFiltersRemoved} filtros,\n Mas o numero de camadas com mais de um filtro e {len(layerSelectedList)}")

    scores = [x for x in zip(allowed_layers, filtersToRemove)]

    if architecture_name.__contains__('ResNet'):
        blocks = rl.count_blocks(model)
        return rebuild_resnet(model=model,
                              blocks=blocks,
                              layer_filters=scores)

    if architecture_name.__contains__('MobileNetV2'):
        blocks = rl.count_blocks(model)
        return rebuild_mobilenetV2(model=model,
                                   blocks=blocks,
                                   layer_filters=scores)

    else:  # If not ResNet nor mobile then it is VGG-Based
        print('TODO: We need to implement (just update) this function')
        return None
