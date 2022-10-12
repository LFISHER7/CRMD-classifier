import os
from keras import models
from keras import layers
from keras.applications import VGG16, VGG19, DenseNet201, InceptionV3
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.layers import Dropout


def assemble(MODEL_NAME, base):
    """Conv base not trainable - feaure extractiion with aug"""
    MODEL_NAME = 'MODEL_NAME'
    tbCallBack = TensorBoard(log_dir='../logs/logs_frozen_conv_bases/{}'.format(MODEL_NAME),
        histogram_freq=0, write_graph=True, write_images=True)

    batch_size = 16
    conv_base = base(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))

    base_dir = '../labels'
    train_dir = os.path.join(base_dir, 'train-equalized')
    test_dir = os.path.join(base_dir, 'test-equalized')


    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(Dropout(0.2))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(13, activation='softmax'))
    model.summary()

    print('This is the number of trainable weights'
	'before freezing the conv base: {}'.format(len(model.trainable_weights)))
    conv_base.trainable = False
    print('This is the number of trainable_weights'
	'after freezing the conv base: {}'.format(len(model.trainable_weights)))


    datagen_train = ImageDataGenerator(rescale=1. / 255,
        rotation_range=180,
        horizontal_flip=True,
        zca_whitening=True,
        height_shift_range=0.2,
        width_shift_range=0.2,
        zoom_range=0.2)

    datagen_test = ImageDataGenerator(rescale=1. / 255)

    generator_train = datagen_train.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',  # useful to useful predct generator
        shuffle=True)

    generator_test = datagen_test.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=True)

    print(generator_train.class_indices)

    model.compile(optimizer=optimizers.Adam(lr=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    training_steps_per_epoch = len(generator_train.filenames) / batch_size
    testing_steps_per_epoch = len(generator_test.filenames) / batch_size

    model.fit_generator(generator_train,
        steps_per_epoch=training_steps_per_epoch,
        epochs=50,
        validation_data=generator_test,
        validation_steps=testing_steps_per_epoch,
        verbose=1,
        callbacks=[tbCallBack])


assemble('VGG16', VGG16)
assemble('VGG19', VGG19)
assemble('DenseNet201', DenseNet201)
assemble('InceptionV3', InceptionV3)
