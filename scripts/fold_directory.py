import os
import glob
from keras import models, layers, optimizers
from keras.applications import VGG16, DenseNet201
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout

CLASSES = [os.path.basename(x) for x in glob.glob("labels_by_fold/0/train/*") if os.path.isdir(x)]
print(f"Classes: {CLASSES}")

def assemble_fold(MODEL_NAME, base, size, block_to_freeze, fold, sequential=True):
    """Running model on all train data with no vaidation.  Fine tuning from block_to_freeze"""

    tbCallBack = TensorBoard(log_dir='./logs/logs_cross_validation/{}-{}'.format(MODEL_NAME, fold),
        histogram_freq=0, write_graph=True, write_images=True)

    batch_size = 16
    conv_base = base(weights='imagenet',
                  include_top=False,
                  input_shape=(size, size, 3))

    base_dir = f'../labels_by_fold/{fold}'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')

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
        x = layers.Activation('relu')(x)
        x = Dropout(0.5)(x)
        predictions = layers.Dense(13, activation='softmax')(x)
        model = models.Model(input=conv_base.input, output=predictions)

    conv_base.trainable = False


    # TRAIN

    datagen_train = ImageDataGenerator(rescale=1. / 255,
        rotation_range=180,
        horizontal_flip=True,
        zca_whitening=True,
        height_shift_range=0.2,
        width_shift_range=0.2,
        zoom_range=0.2)

    #datagen = ImageDataGenerator(rescale=1. / 255)

    generator_train = datagen_train.flow_from_directory(
        train_dir,
        target_size=(size, size),
        batch_size=batch_size,
        classes=CLASSES,
        class_mode='categorical',
        color_mode='rgb',  # useful to useful predct generator
        shuffle=True)

    # VALIDATION

    datagen_test = ImageDataGenerator(rescale=1. / 255,
        rotation_range=180,
        horizontal_flip=True,
        zca_whitening=True,
        height_shift_range=0.2,
        width_shift_range=0.2,
        zoom_range=0.2)

    #datagen = ImageDataGenerator(rescale=1. / 255)

    generator_test = datagen_test.flow_from_directory(
        test_dir,
        target_size=(size, size),
        batch_size=batch_size,
        classes=CLASSES,
        class_mode='categorical',
        color_mode='rgb',  # useful to useful predct generator
        shuffle=True)


    model.compile(optimizer=optimizers.Adam(lr=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    training_steps_per_epoch = len(generator_train.filenames) / batch_size
    testing_steps_per_epoch = len(generator_test.filenames) / batch_size

    model.fit_generator(generator_train,
        steps_per_epoch=training_steps_per_epoch,
        epochs=50,
        verbose=2,
        validation_data=generator_test,
        validation_steps=testing_steps_per_epoch)
    
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
        epochs=50,
        verbose=2,
        callbacks=[tbCallBack],
        validation_data=generator_test,
        validation_steps=testing_steps_per_epoch)

    model.save('../saved_models/model_by_fold_{}_{}'.format(fold, MODEL_NAME))

for fold in range(10):
    print(f"FOLD {fold}")
    assemble_fold('DenseNet201', DenseNet201, 227, 'conv3_block_12_concat', fold, sequential=False)
    assemble_fold('VGG16', VGG16, 150, 'block2_conv1', fold, sequential=True)
