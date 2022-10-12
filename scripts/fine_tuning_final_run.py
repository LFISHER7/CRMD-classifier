import os
from keras import models, layers, optimizers
from keras.applications import VGG16, DenseNet201, InceptionV3
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout

def assemble(MODEL_NAME, base, size, block_to_freeze, sequential=True):
    """Running model on all train data with no vaidation.  Fine tuning from block_to_freeze"""

    tbCallBack = TensorBoard(log_dir='../logs/logs_final_run/{}'.format(MODEL_NAME),
        histogram_freq=0, write_graph=True, write_images=True)

    BATCH_SIZE = 16
    conv_base = base(weights='imagenet',
                  include_top=False,
                  input_shape=(size, size, 3))

    base_dir = '../labels'
    train_dir = os.path.join(base_dir, 'train-equalized')

    if sequential:
        model = models.Sequential()
        model.add(conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(256))
        model.add(layers.Activation('relu'))
        model.add(Dropout(0.5))
        model.add(layers.Dense(13))
        model.add(layers.Activation('softmax'))
    else:
        x = conv_base.output
        x = layers.Flatten()(x)
        x = layers.Dense(256)(x)
        x = BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = Dropout(0.5)(x)
        predictions = layers.Dense(13, activation='softmax')(x)
        model = models.Model(input=conv_base.input, output=predictions)

    model.summary()

    print('This is the number of trainable weights'
    'before freezing the conv base: {}'.format(len(model.trainable_weights)))
    conv_base.trainable =  False
    print('This is the number of trainable_weights'
    'after freezing the conv base: {}'.format(len(model.trainable_weights)))

    datagen_train = ImageDataGenerator(rescale=1. / 255,
        rotation_range=180,
        horizontal_flip=True,
        zca_whitening=True,
        height_shift_range=0.2,
        width_shift_range=0.2,
        zoom_range=0.2)

    generator_train = datagen_train.flow_from_directory(
        train_dir,
        target_size=(size, size),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        color_mode='rgb',  # useful to useful predct generator
        shuffle=True)

    model.compile(optimizer=optimizers.Adam(lr=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    training_steps_per_epoch = len(generator_train.filenames) / BATCH_SIZE
  
    model.fit_generator(generator_train,
        steps_per_epoch=training_steps_per_epoch,
        epochs=100,
        verbose=1)

    conv_base.trainable = True

    set_trainable = False
    for layer in conv_base.layers:
        if layer.name == block_to_freeze:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model.compile(optimizer=optimizers.Adam(lr=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    model.fit_generator(
        generator_train,
        steps_per_epoch=training_steps_per_epoch,
        epochs=100,
        verbose=1,
        callbacks=[tbCallBack])

    model.save('/Users/Louis/Documents/Projects/CRMD_Project/saved_models/njdnfvjndijn{}'.format(MODEL_NAME))

# assemble('VGG16-block4-batch_norm_increasedLR', VGG16, 150, 'block5_conv1', 0.2)

assemble('VGG16', VGG16, 150, 'block4_conv1', sequential=True)
# assemble('DenseNet201', DenseNet201, 227, 'conv3_block_12_concat', 0.5, sequential=False)

# assemble('VGG16-block4_sgd', VGG16, 150, 'block4_conv1', 0.2, SGD)
# assemble('VGG16-block4_sgd', VGG16, 150, 'block4_conv1', 0.2, RMSprop)

# assemble('VGG16-block3_conv1', VGG16, 150, 'block3_conv1', 0.2)
# assemble('VGG16-block2_conv1', VGG16, 150, 'block2_conv1', 0.2)

# assemble('DenseNet201-3', InceptionV3, 227, 'conv3_block_12_concat', 0.2, sequential=False)

#assemble('DenseNet201', DenseNet201, 227, 'conv5_block32_concat', 0.2)


# build(DenseNet201, 'Densenet201-baseline', 'conv5_block32_2_conv', 227, 16, 'no')
# build(DenseNet201, 'Densenet201-more_unfrozen', 'conv5_block32_1_conv', 227, 16, 'no')
# build(DenseNet201, 'Densenet201-block31', 'conv5_block31_1_conv', 227, 16, 'no')
# build(VGG16, 'VGG16-baseline', 'block5_conv1', 150, 16, 'no')
# build(VGG16, 'VGG16-more_unfrozen', 'block4_conv1', 150, 16, 'no')
# build(VGG16, 'VGG16-block3', 'block3_conv1', 150, 16, 'no')

# assemble('DenseNet201', DenseNet201, 227, 'conv5_block32_concat')
