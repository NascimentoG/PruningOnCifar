import numpy as np
import random
import time
import tensorflow as tf
from tensorflow.keras import layers, backend as K, Input, Model
from tensorflow.keras.utils import custom_object_scope, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# -------------------------------
# Transformer Utilities
# -------------------------------

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super(Patches, self).__init__()
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
        config = super().get_config().copy()
        config.update({'patch_size': self.patch_size})
        return config

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=self.projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches, output_dim=self.projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

    def get_config(self):
        config = super().get_config().copy()
        config.update({'num_patches': self.num_patches, 'projection_dim': self.projection_dim})
        return config

def load_transformer_model(architecture_file='', weights_file=''):
    import tensorflow.keras as keras
    if '.json' not in architecture_file:
        architecture_file += '.json'

    with open(architecture_file, 'r') as f:
        with custom_object_scope({'PatchEncoder': PatchEncoder, 'Patches': Patches}):
            model = keras.models.model_from_json(f.read())

    if weights_file != '':
        if '.h5' not in weights_file:
            weights_file += '.h5'
        model.load_weights(weights_file)
        print(f'Load architecture [{architecture_file}]. Load weights [{weights_file}]', flush=True)
    else:
        print(f'Load architecture [{architecture_file}]', flush=True)

    return model

# -------------------------------
# General Model Utilities
# -------------------------------

def _hard_swish(x):
    return x * K.relu(x + 3.0, max_value=6.0) / 6.0

def _relu6(x):
    return K.relu(x, max_value=6.0)

def load_model(architecture_file='', weights_file=''):
    import tensorflow.keras as keras
    if '.json' not in architecture_file:
        architecture_file += '.json'

    with open(architecture_file, 'r') as f:
        with custom_object_scope({'relu6': _relu6, 'DepthwiseConv2D': layers.DepthwiseConv2D, '_hard_swish': _hard_swish}):
            model = keras.models.model_from_json(f.read())

    if weights_file != '':
        if '.h5' not in weights_file:
            weights_file += '.h5'
        model.load_weights(weights_file)
        print(f'Load architecture [{architecture_file}]. Load weights [{weights_file}]', flush=True)
    else:
        print(f'Load architecture [{architecture_file}]', flush=True)

    return model

def save_model(file_name='', model=None):
    import tensorflow.keras as keras
    print(f'Salving architecture and weights in {file_name}')
    model.save_weights(file_name + '.h5')
    with open(file_name + '.json', 'w') as f:
        f.write(model.to_json())

# -------------------------------
# Dataset Utilities
# -------------------------------

def cifar_resnet_data(debug=True, validation_set=False, n_samples=10):
    from tensorflow.keras.datasets import cifar10
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

    if debug:
        n_classes = len(np.unique(y_train, axis=0))
        print(f'Debuging Mode. Sampling {n_samples*n_classes} samples ({n_samples} for each classe)')
        n_samples = n_samples * n_classes
        y_ = np.argmax(y_train, axis=1)
        sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in np.unique(y_)]
        x_train = x_train[np.array(sub_sampling).reshape(-1)]
        y_train = y_train[np.array(sub_sampling).reshape(-1)]
        y_ = np.argmax(y_test, axis=1)
        sub_sampling = [np.random.choice(np.where(y_ == value)[0], n_samples, replace=False) for value in np.unique(y_)]
        x_test = x_test[np.array(sub_sampling).reshape(-1)]
        y_test = y_test[np.array(sub_sampling).reshape(-1)]

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    if not validation_set:
        return x_train, y_train, x_test, y_test
    else:
        datagen = generate_data_augmentation(x_train)
        for x_val, y_val in datagen.flow(x_train, y_train, batch_size=5000):
            break
        return x_train, y_train, x_test, y_test, x_val, y_val

def image_net_data(load_train=True, load_test=True, subtract_pixel_mean=False,
                   path='', train_size=1.0, test_size=1.0):
    X_train, y_train, X_test, y_test = (None, None, None, None)
    if load_train:
        tmp = np.load(path+'imagenet_train.npz')
        X_train, y_train = tmp['X'], tmp['y']
        if train_size != 1.0:
            X_train, _, y_train, _ = train_test_split(X_train, y_train, random_state=42, train_size=train_size)
        X_train = X_train.astype('float32') / 255
        y_train = to_categorical(y_train, 1000)

    if load_test:
        tmp = np.load(path+'imagenet_val.npz')
        X_test, y_test = tmp['X'], tmp['y']
        if test_size != 1.0:
            _, X_test, _, y_test = train_test_split(X_test, y_test, random_state=42, test_size=test_size)
        X_test = X_test.astype('float32') / 255
        y_test = to_categorical(y_test, 1000)

    if subtract_pixel_mean:
        X_train_mean = np.load(path + 'x_train_mean.npz')['X']
        if load_train:
            X_train -= X_train_mean
        if load_test:
            X_test -= X_train_mean

    print(f'#Training Samples [{0 if X_train is None else X_train.shape[0]}]')
    print(f'#Testing Samples [{0 if X_test is None else X_test.shape[0]}]')
    return X_train, y_train, X_test, y_test

def image_net_tiny_data(subtract_pixel_mean=False,
                   path='../../datasets/ImageNetTiny/TinyImageNet.npz', train_size=1.0, test_size=1.0):
    tmp = np.load(path)
    X_train, y_train, X_test, y_test = tmp['X_train'], tmp['y_train'], tmp['X_test'], tmp['y_test']
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    y_train = to_categorical(y_train, 200)
    y_test = to_categorical(y_test, 200)

    if train_size != 1.0:
        X_train, _, y_train, _ = train_test_split(X_train, y_train, random_state=42, train_size=train_size)
    if test_size != 1.0:
        _, X_test, _, y_test = train_test_split(X_test, y_test, random_state=42, test_size=test_size)

    if subtract_pixel_mean:
        X_train_mean = np.mean(X_train, axis=0)
        X_train -= X_train_mean
        X_test -= X_train_mean

    print(f'#Training Samples [{X_train.shape[0]}]')
    print(f'#Testing Samples [{X_test.shape[0]}]')
    return X_train, y_train, X_test, y_test

# -------------------------------
# Optimizer & Learning Rate
# -------------------------------

def optimizer_compile(model, model_type='VGG16'):
    import tensorflow.keras as keras
    if model_type == 'VGG16':
        sgd = keras.optimizers.SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    elif model_type == 'ResNetV1':
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=1e-3), metrics=['accuracy'])
    return model

def lr_schedule(epoch, init_lr=0.01, schedule=[(25, 0.001), (50, 0.0001)]):
    for i in range(len(schedule)-1):
        if schedule[i][0] < epoch <= schedule[i+1][0]:
            print('Learning rate:', schedule[i][1])
            return schedule[i][1]
    if epoch > schedule[-1][0]:
        print('Learning rate:', schedule[-1][1])
        return schedule[-1][1]
    print('Learning rate:', init_lr)
    return init_lr

def generate_data_augmentation(X_train):
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
    )
    datagen.fit(X_train)
    return datagen

# -------------------------------
# Model Metrics & Utilities
# -------------------------------

def count_filters(model):
    n_filters = 0
    for layer in model.layers[1:]:
        if isinstance(layer, layers.Conv2D) and not isinstance(layer, layers.DepthwiseConv2D):
            n_filters += layer.filters
        elif isinstance(layer, layers.DepthwiseConv2D):
            n_filters += layer.output_shape[-1]
    return n_filters

def count_depth(model):
    depth = sum([1 for layer in model.layers if isinstance(layer, layers.Conv2D)])
    print(f'Depth: [{depth}]')
    return depth

def meanLatency(model, X_test):
    start = time.time()
    model.predict(X_test)
    return (time.time() - start) / len(X_test)

def top_k_accuracy(y_true, y_pred, k):
    top_n = np.argsort(y_pred, axis=1)[:,-k:]
    idx_class = np.argmax(y_true, axis=1)
    hit = sum([1 if idx_class[i] in top_n[i,:] else 0 for i in range(idx_class.shape[0])])
    return float(hit)/idx_class.shape[0]

# -------------------------------
# Image Preprocessing
# -------------------------------

def center_crop(image, crop_size=224):
    h, w, _ = image.shape
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return image[top:top+crop_size, left:left+crop_size, :]

def random_crop(img, random_crop_size=(64, 64)):
    height, width = img.shape[:2]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:y+dy, x:x+dx, :]

def data_augmentation_images(X, padding=4):
    X_out = np.zeros_like(X)
    n_samples, x, y, _ = X.shape
    padded_sample = np.zeros((x+2*padding, y+2*padding, 3), dtype=X.dtype)
    for i in range(n_samples):
        p = random.random()
        padded_sample[padding:x+padding, padding:y+padding, :] = X[i]
        if p >= 0.5:
            X_out[i] = random_crop(padded_sample, (x, y))
        else:
            X_out[i] = random_crop(np.flip(padded_sample, axis=1), (x, y))
    return X_out

def cutout(img):
    MAX_CUTS = 5
    MAX_LENGTH_MULTIPLIER = 5
    height, width, channels = img.shape
    mask = np.ones_like(img, dtype=np.float32)
    nb_cuts = np.random.randint(0, MAX_CUTS+1)
    for _ in range(nb_cuts):
        y = np.random.randint(height)
        x = np.random.randint(width)
        length = 4 * np.random.randint(1, MAX_LENGTH_MULTIPLIER+1)
        y1, y2 = np.clip([y - length//2, y + length//2], 0, height)
        x1, x2 = np.clip([x - length//2, x + length//2], 0, width)
        mask[y1:y2, x1:x2, :] = 0.
    return img * mask

def memory_usage(batch_size, model):
    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None or not isinstance(s, int):
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem
    trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
    non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])
    number_size = {'float16':2., 'float32':4., 'float64':8.}[K.floatx()]
    total_memory = number_size*(batch_size*shapes_mem_count + trainable_count + non_trainable_count)
    return total_memory / (1024.**2)

# -------------------------------
# Model Generation
# -------------------------------

def generate_conv_model(model_name='VGG16', input_shape=(32,32,3), n_classes=10, depth_multiplier=1):
    def conv_block(x, filters, kernel_size=3, stride=1, use_bn=True):
        x = layers.Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same', use_bias=not use_bn)(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def residual_block(x, filters, use_bn=True):
        shortcut = x
        x = conv_block(x, filters, kernel_size=3, stride=1, use_bn=use_bn)
        x = conv_block(x, filters, kernel_size=3, stride=1, use_bn=use_bn)
        x = layers.Add()([shortcut, x])
        x = layers.Activation('relu')(x)
        return x

    inputs = Input(shape=input_shape)
    x = inputs

    if model_name.lower() == 'vgg16':
        filter_list = [64, 128, 256, 512, 512]
        for filters in filter_list:
            filters = int(filters * depth_multiplier)
            x = conv_block(x, filters)
            x = conv_block(x, filters)
            x = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
        x = layers.Flatten()(x)
        x = layers.Dense(int(4096*depth_multiplier), activation='relu')(x)
        x = layers.Dense(int(4096*depth_multiplier), activation='relu')(x)
        outputs = layers.Dense(n_classes, activation='softmax')(x)

    elif model_name.lower() == 'resnet':
        x = conv_block(x, 64, kernel_size=7, stride=2)
        x = layers.MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)
        filters_list = [64, 128, 256, 512]
        n_blocks_list = [2, 2, 2, 2]
        for filters, n_blocks in zip(filters_list, n_blocks_list):
            filters = int(filters * depth_multiplier)
            for _ in range(n_blocks):
                x = residual_block(x, filters)
            if filters != filters_list[-1]:
                x = conv_block(x, filters*2, kernel_size=1, stride=2)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(n_classes, activation='softmax')(x)

    else:
        raise ValueError(f'Model [{model_name}] not implemented')

    model = Model(inputs, outputs)
    return model
