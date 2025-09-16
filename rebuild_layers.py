import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils import gen_batches
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import template_architectures
import copy
import gc

architecture_name = 'ResNet'


def filters_layerResNet(model, mask):
    output = []

    # Add the same weights until finding the first Add layer
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Conv2D):
            output.append((i, int(layer.output.shape[-1])))

        if isinstance(layer, Add):
            break

    add_model = add_to_pruneResNet(model)
    add_model = np.array(add_model)[mask == 1]
    add_model = list(add_model)

    for layer_idx in range(len(add_model)):
        idx_model = np.arange(add_model[layer_idx] - 6, add_model[layer_idx] + 1).tolist()
        for i in idx_model:
            layer = model.get_layer(index=i)
            if isinstance(layer, Conv2D):
                output.append((i, int(layer.output.shape[-1])))

    add_model = add_to_downsampling(model)
    for layer_idx in range(len(add_model)):
        idx_model = np.arange(add_model[layer_idx] - 7, add_model[layer_idx] + 1).tolist()
        for i in idx_model:
            layer = model.get_layer(index=i)
            if isinstance(layer, Conv2D):
                output.append((i, int(layer.output.shape[-1])))

    output.sort(key=lambda tup: tup[0])
    output = [item[1] for item in output]
    return output


def filters_layerMobileNetV2(model, mask):
    output = []

    # Add the same weights until finding the first block_ layer
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Conv2D) and not isinstance(layer, DepthwiseConv2D):
            output.append((i, int(layer.output.shape[-1])))

        if 'block_' in layer.name:
            break

    add_model = add_to_pruneMobileV2(model)
    add_model = np.array(add_model)[mask == 1]
    add_model = list(add_model)

    for layer_idx in range(len(add_model)):
        idx_model = np.arange(add_model[layer_idx] - 8, add_model[layer_idx] + 1).tolist()
        for i in idx_model:
            layer = model.get_layer(index=i)
            if isinstance(layer, Conv2D) and not isinstance(layer, DepthwiseConv2D):
                output.append((i, int(layer.output.shape[-1])))

    add_model = remain_layers_MobileNetV2(model)
    for i in add_model:
        layer = model.get_layer(index=i)
        if isinstance(layer, Conv2D) and not isinstance(layer, DepthwiseConv2D):
            output.append((i, int(layer.output.shape[-1])))

    output = list(set(output))  # Remove duplicates
    output.sort(key=lambda tup: tup[0])
    output = [item[1] for item in output]
    return output


def add_to_pruneResNet(model):
    allowed_layers = []
    all_add = []

    for i in range(len(model.layers)):
        if isinstance(model.get_layer(index=i), Add):
            all_add.append(i)

    for i in range(1, len(all_add) - 1):
        input_shape = model.get_layer(index=all_add[i]).output.shape
        output_shape = model.get_layer(index=all_add[i - 1]).output.shape
        # These are the valid blocks we can remove
        if input_shape == output_shape:
            allowed_layers.append(all_add[i])

    # The last block is enabled
    if len(all_add) > 0:
        allowed_layers.append(all_add[-1])
    return allowed_layers


def add_to_pruneMobileV2(model):
    allowed_layers = []
    all_add = []

    for i in range(len(model.layers)):
        if isinstance(model.get_layer(index=i), Add):
            all_add.append(i)

    for i in range(1, len(all_add)):
        input_shape = model.get_layer(index=all_add[i]).output.shape
        output_shape = model.get_layer(index=all_add[i - 1]).output.shape
        # These are the valid blocks we can remove
        if input_shape == output_shape:
            allowed_layers.append(all_add[i])

    # The last block is always enabled (commented out in original)
    # if len(all_add) > 0:
    #     allowed_layers.append(all_add[-1])
    return allowed_layers


def add_to_downsampling(model):
    layers = []
    all_add = []

    for i in range(len(model.layers)):
        if isinstance(model.get_layer(index=i), Add):
            all_add.append(i)

    for i in range(1, len(all_add) - 1):
        input_shape = model.get_layer(index=all_add[i]).output.shape
        output_shape = model.get_layer(index=all_add[i - 1]).output.shape
        # These are the downsampling add
        if input_shape != output_shape:
            layers.append(all_add[i])

    return layers


def idx_score_block_ResNet(blocks, layers):
    # Associates the scores' index with the ResNet block
    output = {}
    idx = 0
    for i in range(len(blocks)):
        for layer_idx in range(idx, idx + blocks[i] - 1):
            output[layers[layer_idx]] = i
            idx = idx + 1
    return output


def idx_score_block_MobileV2(model, layers):
    output = {}
    shapes = [24, 32, 64, 96, 160]  # Shapes (represented by #filters) of the blocks

    for i in range(len(layers)):
        shape = int(model.get_layer(index=layers[i]).output.shape[-1])
        idx = shapes.index(shape)
        output[layers[i]] = idx

    return output


def new_blocks(blocks, scores, allowed_layers, score_block, p=0.1):
    num_blocks = blocks

    if isinstance(p, float):
        num_remove = round(p * len(scores))
    else:
        num_remove = p
    mask = np.ones(len(allowed_layers), dtype=int)

    # It forces to remove 'num_remove' layers
    i = num_remove
    scores = np.array(scores, copy=True)
    while i > 0 and not np.all(np.isinf(scores)):
        min_score = int(np.argmin(scores))  # Finds the minimum VIP
        block_layer_index = allowed_layers[min_score]  # layer index associated with min vip
        block_idx = score_block[block_layer_index]
        if num_blocks[block_idx] - 1 > 1:
            mask[min_score] = 0
            num_blocks[block_idx] = num_blocks[block_idx] - 1
            i = i - 1

        scores[min_score] = np.inf  # Removes the minimum VIP from the list

    return num_blocks, mask


def transfer_weightsResNetBN(model, new_model, mask):
    add_model = add_to_pruneResNet(model)
    add_new_model = add_to_pruneResNet(new_model)

    # Add the same weights until finding the first Add layer
    for idx in range(len(model.layers)):
        w = model.get_layer(index=idx).get_weights()
        new_model.get_layer(index=idx).set_weights(w)

        if isinstance(model.get_layer(index=idx), Add):
            break

    # These are the layers where the weights must be transferred
    add_model = np.array(add_model)[mask == 1]
    add_model = list(add_model)
    end = len(add_new_model)

    for layer_idx in range(end):
        idx_model = np.arange(add_model[0] - 9, add_model[0] + 1).tolist()
        idx_new_model = np.arange(add_new_model[0] - 9, add_new_model[0] + 1).tolist()

        for transfer_idx in range(len(idx_model)):
            w = model.get_layer(index=idx_model[transfer_idx]).get_weights()
            new_model.get_layer(index=idx_new_model[transfer_idx]).set_weights(w)

        add_new_model.pop(0)
        add_model.pop(0)

    # These are the downsampling layers
    add_model = add_to_downsampling(model)
    add_new_model = add_to_downsampling(new_model)

    for i in range(len(add_model)):
        idx_model = np.arange(add_model[i] - 11, add_model[i] + 1).tolist()
        idx_new_model = np.arange(add_new_model[i] - 11, add_new_model[i] + 1).tolist()

        for transfer_idx in range(len(idx_model)):
            w = model.get_layer(index=idx_model[transfer_idx]).get_weights()
            new_model.get_layer(index=idx_new_model[transfer_idx]).set_weights(w)

    # This is the dense layer
    w = model.get_layer(index=-1).get_weights()
    new_model.get_layer(index=-1).set_weights(w)

    return new_model


def transfer_weightsResNet(model, new_model, mask):
    add_model = add_to_pruneResNet(model)
    add_new_model = add_to_pruneResNet(new_model)

    # Add the same weights until finding the first Add layer
    for idx in range(len(model.layers)):
        w = model.get_layer(index=idx).get_weights()
        new_model.get_layer(index=idx).set_weights(w)

        if isinstance(model.get_layer(index=idx), Add):
            break

    # These are the layers where the weights must be transferred
    add_model = np.array(add_model)[mask == 1]
    add_model = list(add_model)
    end = len(add_new_model)

    for layer_idx in range(end):
        idx_model = np.arange(add_model[0] - 6, add_model[0] + 1).tolist()
        idx_new_model = np.arange(add_new_model[0] - 6, add_new_model[0] + 1).tolist()

        for transfer_idx in range(len(idx_model)):
            w = model.get_layer(index=idx_model[transfer_idx]).get_weights()
            new_model.get_layer(index=idx_new_model[transfer_idx]).set_weights(w)

        add_new_model.pop(0)
        add_model.pop(0)

    # These are the downsampling layers
    add_model = add_to_downsampling(model)
    add_new_model = add_to_downsampling(new_model)

    for i in range(len(add_model)):
        idx_model = np.arange(add_model[i] - 7, add_model[i] + 1).tolist()
        idx_new_model = np.arange(add_new_model[i] - 7, add_new_model[i] + 1).tolist()

        for transfer_idx in range(len(idx_model)):
            w = model.get_layer(index=idx_model[transfer_idx]).get_weights()
            new_model.get_layer(index=idx_new_model[transfer_idx]).set_weights(w)

    # This is the dense layer
    w = model.get_layer(index=-1).get_weights()
    new_model.get_layer(index=-1).set_weights(w)

    return new_model


def transfer_weightsMobileNetV2(model, new_model, mask):
    add_model = add_to_pruneMobileV2(model)
    add_new_model = add_to_pruneMobileV2(new_model)

    assigned_weights = np.zeros((len(new_model.layers)), dtype=bool)

    # Add the same weights until finding the first block_ layer (Expand layer)
    for idx in range(len(model.layers)):
        w = model.get_layer(index=idx).get_weights()
        new_model.get_layer(index=idx).set_weights(w)

        if 'block_' in model.get_layer(index=idx).name:
            break

        assigned_weights[idx] = True

    # These are the layers where the weights must be transferred
    add_model = np.array(add_model)[mask == 1]
    add_model = list(add_model)
    end = len(add_new_model)

    for layer_idx in range(end):
        idx_model = np.arange(add_model[0] - 8, add_model[0] + 1).tolist()
        idx_new_model = np.arange(add_new_model[0] - 8, add_new_model[0] + 1).tolist()

        for transfer_idx in range(len(idx_model)):
            w = model.get_layer(index=idx_model[transfer_idx]).get_weights()
            new_model.get_layer(index=idx_new_model[transfer_idx]).set_weights(w)
            assigned_weights[idx_new_model[transfer_idx]] = True

        add_new_model.pop(0)
        add_model.pop(0)

    # These are the downsampling and remaining layers
    idx_model = remain_layers_MobileNetV2(model)
    idx_new_model = remain_layers_MobileNetV2(new_model)

    for i in range(len(idx_new_model)):
        w = model.get_layer(index=idx_model[i]).get_weights()
        new_model.get_layer(index=idx_new_model[i]).set_weights(w)
        assigned_weights[idx_new_model[i]] = True

    for i in range(len(assigned_weights)):
        if not assigned_weights[i]:
            print(f'Caution! Weights from Layer[{i}] were not transferred')

    return new_model


def remain_layers_MobileNetV2(model):
    layers = []
    last_i = 0

    for i in range(len(model.layers)):
        name = model.get_layer(index=i).name

        if '_pad' in name:
            layers.append(np.arange(i - 3, i))
            layers.append(np.arange(i, i + 15))

        if '_add' in name:
            n_filters = int(model.get_layer(index=i).output.shape[-1])
            if n_filters == 96:  # E.g., layer name = 'block_10_expand'
                dist = 1
                # We need a more elegant solution, keep original heuristic
                while not ('_add' in model.get_layer(index=i - dist).name):
                    dist = dist + 1

                if dist != 9:
                    layers.append(np.arange(i - 16, i + 1))
        last_i = i

    # Classification layers (Head) - Logits
    layers.append(np.arange(last_i - 12, last_i + 1))
    # flatten lists of ranges into single list
    layers = [item for sublist in layers for item in sublist]
    return layers


def count_res_blocks(model, dim_block=[16, 32, 64]):
    # Returns the last Add of each block
    res_blocks = [0, 0, 0]

    for i in range(len(model.layers) - 1):
        layer = model.get_layer(index=i)
        if isinstance(layer, Add):
            dim = int(layer.output.shape[-1])
            if dim == 16:
                res_blocks[0] += 1
            elif dim == 32:
                res_blocks[1] += 1
            elif dim == 64:
                res_blocks[2] += 1

    return res_blocks


def count_mobilev2_blocks(model):
    shapes = [24, 32, 64, 96, 160]
    add_layers = []

    for i in range(len(model.layers)):
        if isinstance(model.get_layer(index=i), Add):
            add_layers.append(i)

    blocks = np.ones((len(shapes)), dtype=int)

    for i in range(len(add_layers)):
        shape = int(model.get_layer(index=add_layers[i]).output.shape[-1])
        idx = shapes.index(shape)
        blocks[idx] = blocks[idx] + 1

    return blocks


def count_blocks(model):
    if 'ResNet' in architecture_name:
        return count_res_blocks(model)
    if 'MobileNetV2' in architecture_name:
        return count_mobilev2_blocks(model)


def blocks_to_prune(model):
    if 'ResNet' in architecture_name:
        return add_to_pruneResNet(model)
    if 'MobileNetV2' in architecture_name:
        return add_to_pruneMobileV2(model)


def rebuild_network(model, scores, p_layer):
    allowed_layers = [x[0] for x in scores]
    scores = [x[1] for x in scores]

    if 'ResNet' in architecture_name:
        filters_layers = filters_layerResNet
        create_model = template_architectures.ResNet
        transfer_weights = transfer_weightsResNet
        blocks = count_res_blocks(model)
        score_block = idx_score_block_ResNet(blocks, allowed_layers)

    if 'MobileNetV2' in architecture_name:
        filters_layers = filters_layerMobileNetV2
        create_model = template_architectures.MobileNetV2
        transfer_weights = transfer_weightsMobileNetV2
        blocks = count_mobilev2_blocks(model)
        score_block = idx_score_block_MobileV2(model, allowed_layers)

    blocks_tmp, mask = new_blocks(blocks, scores, allowed_layers, score_block, p=p_layer)
    filters = filters_layers(model, mask)
    tmp_model = create_model(input_shape=(32, 32, 3),
                             depth_block=copy.deepcopy(blocks_tmp),
                             filters=copy.deepcopy(filters),
                             num_classes=10)
    pruned_model_layer = transfer_weights(model, tmp_model, mask)

    return pruned_model_layer
