# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + _cell_guid="b1076dfc-b9ad-4769-8c92-a6c4dae69d19" _uuid="8f2839f25d086af736a60e9eeb907d3b93b6e0e5" pycharm={"name": "#%%\n"}
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Dense, Input, Softmax, BatchNormalization, Conv2D, MaxPool2D, 
                                     AveragePooling2D, Add, Flatten, Dropout, Activation, ZeroPadding2D)
from tensorflow.keras.optimizers import Adam

# + pycharm={"name": "#%%\n"}
# Load data:

data = tf.keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = data

# + pycharm={"name": "#%%\n"}
print('x_train shape:', x_train.shape)
print('x_teset shape:', x_test.shape)

m_train_examples = x_train.shape[0]
m_test_examples = x_test.shape[0]

print('# of train examples:', m_train_examples)
print('# of test examples:', m_test_examples)

image_shape = x_train.shape[1:]
print('image shape:', image_shape)

# + pycharm={"name": "#%%\n"}
LABEL_NAMES = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

NUM_CLASSES = len(LABEL_NAMES)

# + pycharm={"name": "#%%\n"}
# Process Data:
# TODO: we subtract TRAIN mean and divide by TRAIN std on the TEST distribution as well -- why?
mean = np.mean(x_train)
std = np.std(x_train)

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# + pycharm={"name": "#%%\n"}
# convert to one-hot encoding:
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# + pycharm={"name": "#%%\n"}
# Explore data:

RAND_INDEX = np.random.randint(m_train_examples)
label = np.argmax(y_train[RAND_INDEX]) # we use `argmax` since we converted the labels from list of ints to one-hot encoding

img = x_train[RAND_INDEX]
img_title = 'label: %s (%s)' % (LABEL_NAMES[label], label)

plt.imshow(img)
plt.title(img_title)
plt.axis('off')
plt.show()

# + pycharm={"name": "#%%\n"}
# Setup data generator:

BATCH_SIZE = 128

datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

datagen.fit(x_train)

train_generator = datagen.flow(
    x_train,
    y_train,
    batch_size=BATCH_SIZE
)


# + pycharm={"name": "#%%\n"}
def identity_block(input_tensor, kernel_size, num_filters, stage_label, block_label, print_dimensions=True):
    """ Standard block in ResNet -- corresponds to when input activation (e.g. a[l]) has same dimension as output activation
        a[l+2].
        
        Note: Identity blocks' shortcuts (skip connections) are parameter-free (since they're simply adding input tensors)
    """
    
    num_filters_1, num_filters_2, num_filters_3 = num_filters
    CONV_BASE_NAME = 'conv_%s_%s' % (stage_label, block_label)
    BATCH_NORM_BASE_NAME = 'batch_norm_%s_%s' % (stage_label, block_label)
    
    x = Conv2D(num_filters_1, (1, 1), padding='valid', name=CONV_BASE_NAME+'a')(input_tensor) # strides = (1, 1) which is the default arg
    x = BatchNormalization(axis=-1, name=BATCH_NORM_BASE_NAME+'a')(x) # axis should be set to features
    x = Activation('relu')(x)
    
    x = Conv2D(num_filters_2, kernel_size, padding='same', name=CONV_BASE_NAME+'b')(x)
    x = BatchNormalization(axis=-1, name=BATCH_NORM_BASE_NAME+'b')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(num_filters_3, (1, 1), padding='valid', name=CONV_BASE_NAME+'c')(x)
    x = BatchNormalization(axis=-1, name=BATCH_NORM_BASE_NAME+'c')(x)
    
    if print_dimensions:
        print('------- %s - identity block -------' % stage_label)
        print('input_tensor shape:', input_tensor.shape)
        print('x shape:', x.shape, end='\n\n')
    
    x = tf.keras.layers.add([x, input_tensor]) # inject skip connection
    x = Activation('relu')(x)
    
    return x

def conv_block(input_tensor, kernel_size, num_filters, stage_label, block_label, strides=(2, 2), print_dimensions=True):
    """ The conv block is used when input and output have different dimensions.
    
        The conv black differs from identity block by having convolution layer in the skip connection;
        The conv layer in the shortcut is used to resize the input to different dimension so that the dimensions 
        match when the shortcut is added (tf.keras.layers.add) back to the main path.
    """
    
    num_filters_1, num_filters_2, num_filters_3 = num_filters
    CONV_BASE_NAME = 'conv_%s_%s' % (stage_label, block_label)
    BATCH_NORM_BASE_NAME = 'batch_norm_%s_%s' % (stage_label, block_label)
    
    # if strides != 1, then we are downsampling/changing dimensions
    # in standard practice, the image dimensions decrease in later layers, so this is ok
    x = Conv2D(num_filters_1, (1, 1), strides=strides, padding='valid', name=CONV_BASE_NAME+'a')(input_tensor)
    x = BatchNormalization(axis=-1, name=BATCH_NORM_BASE_NAME+'a')(x) # axis should be set to features
    x = Activation('relu')(x)
    
    x = Conv2D(num_filters_2, kernel_size, padding='same', name=CONV_BASE_NAME+'b')(x)
    x = BatchNormalization(axis=-1, name=BATCH_NORM_BASE_NAME+'b')(x)
    x = Activation('relu')(x)
    
    x = Conv2D(num_filters_3, (1, 1), padding='valid', name=CONV_BASE_NAME+'c')(x)
    x = BatchNormalization(axis=-1, name=BATCH_NORM_BASE_NAME+'c')(x)
    x = Activation('relu')(x)
    
    shortcut = Conv2D(num_filters_3, (1, 1), strides=strides, padding='valid', name=CONV_BASE_NAME+'shortcut')(input_tensor)
    shortcut = BatchNormalization(axis=-1, name=BATCH_NORM_BASE_NAME+'shortcut')(shortcut)
        
    if print_dimensions:
        print('------- %s - conv block -------' % stage_label)
        print('input_tensor shape:', input_tensor.shape)
        print('x shape:', x.shape)
        print('shortcut shape:', shortcut.shape, end='\n\n')
    
    x = tf.keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    
    return x


# + pycharm={"name": "#%%\n"}
# Model:

num_classes=6

i = Input(shape=image_shape)
x = ZeroPadding2D((3, 3))(i)
x = Conv2D(64, (7, 7), strides=(2, 2), name='conv_1')(x)
x = BatchNormalization(name='bn_conv_1')(x) # axis=-1 is already default
x = Activation('relu')(x)
x = MaxPool2D((3, 3), strides=(2, 2), name='max_pool_1')(x)

# n.b. the # of filters passed to each res block is important -- if it's not correct, 
# we will get dimension mismatch (specifically, when trying to add the skip connection)
x = conv_block(x, 3, [64, 64, 256], stage_label=2, block_label='a', strides=(1, 1))
x = identity_block(x, 3, [64, 64, 256], stage_label=2, block_label='b')
x = identity_block(x, 3, [64, 64, 256], stage_label=2, block_label='c')

x = conv_block(x, 3, [128, 128, 512], stage_label=3, block_label='a', strides=(2, 2))
x = identity_block(x, 3, [128, 128, 512], stage_label=3, block_label='b')
x = identity_block(x, 3, [128, 128, 512], stage_label=3, block_label='c')
x = identity_block(x, 3, [128, 128, 512], stage_label=3, block_label='d')

x = conv_block(x, 3, [256, 256, 1024], stage_label=4, block_label='a', strides=(2, 2))
x = identity_block(x, 3, [256, 256, 1024], stage_label=4, block_label='b')
x = identity_block(x, 3, [256, 256, 1024], stage_label=4, block_label='c')
x = identity_block(x, 3, [256, 256, 1024], stage_label=4, block_label='d')
x = identity_block(x, 3, [256, 256, 1024], stage_label=4, block_label='e')
x = identity_block(x, 3, [256, 256, 1024], stage_label=4, block_label='f')

x = conv_block(x, 3, [512, 512, 2048], stage_label=5, block_label='a', strides=(2, 2))
x = identity_block(x, 3, [512, 512, 2048], stage_label=5, block_label='b')
x = identity_block(x, 3, [512, 512, 2048], stage_label=5, block_label='c')

# x = AveragePooling2D((2, 2), name='avg_pool')(x)
x = Flatten()(x)
x = Dense(NUM_CLASSES, activation='softmax', name='fully_connected')(x)

model = Model(i, x)

# + pycharm={"name": "#%%\n"}
model.compile(
    optimizer=Adam(lr=0.001,decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model_history = model.fit(
    train_generator,
    steps_per_epoch = m_train_examples // BATCH_SIZE,
    epochs=100,
    validation_data=(x_test, y_test)
)

# + [markdown] pycharm={"name": "#%% md\n"}
# ## Model Summary:

# + pycharm={"name": "#%%\n"}
model.summary()

# + pycharm={"name": "#%%\n"}
plt.plot(model_history.history['accuracy'], label='accuracy')
plt.plot(model_history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()

plt.plot(model_history.history['loss'], label='loss')
plt.plot(model_history.history['val_loss'], label='val_loss')
plt.legend()
plt.title('Loss')
plt.show()
# + pycharm={"name": "#%%\n"}

