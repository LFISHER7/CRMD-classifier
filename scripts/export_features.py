import os
import numpy as np
from keras.applications import VGG16, VGG19, InceptionV3, InceptionResNetV2, DenseNet121, Xception, DenseNet201
from keras.preprocessing.image import ImageDataGenerator

def instantiate_base(model, size):
    """Creates convolutional base, pretrained on Imagenet.  No top"""
    conv_base = model(weights='imagenet',
                  include_top=False, #Replace top with own classifier
                  input_shape=(size, size, 3))
    conv_base.summary()
    return conv_base

def train(name, datagen, dir, batch_size, conv_base, size):
    """Run images through conv base and make prediction.  Save outut.
    Training"""
    generator_train = datagen.flow_from_directory(
        dir,
        target_size=(size, size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb',  # useful to useful predct generator
        shuffle=False)

    bottleneck_features_train = conv_base.predict_generator(generator_train, 800) #800 train samples
    labels = [filename.split('/', 1)[0] for filename in generator_train.filenames]

    print('Bottleneck shape: {}'.format(bottleneck_features_train.shape))
    print('{} labels'.format(len(labels)))

    with open('../different_model_weights/train_{}.npy'.format(name), 'wb') as f:
        np.save(f, bottleneck_features_train)

def test(name, datagen, dir, batch_size, conv_base, size):
    """Run images through conv base and make prediction.  Save outut.
    Testing"""
    generator_test = datagen.flow_from_directory(
        dir,
        target_size=(size, size),
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='rgb', # useful to useful predct generator
        shuffle=False)

    bottleneck_features_test = conv_base.predict_generator(generator_test, 163) #163 test samples
    labels = [filename.split('\\', 1)[0] for filename in generator_test.filenames]

    print('Bottleneck shape: {}'.format(bottleneck_features_test.shape))
    print('{} labels'.format(len(labels)))

    with open('../different_model_weights/test_{}.npy'.format(name), 'wb') as f:
        np.save(f, bottleneck_features_test)

def build(base, name, size):
    """Building model and getting predictions"""
    conv_base = instantiate_base(base, size)
    base_dir = '../labels'
    datagen = ImageDataGenerator(rescale=1. / 255)
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    batch_size=1
    train(name, datagen, train_dir, batch_size, conv_base, size)
    test(name, datagen, test_dir, batch_size, conv_base, size)

# build(VGG16, 'VGG16', 150)
# build(VGG19, 'VGG19', 150)
# build(InceptionV3, 'InceptionV3', 299)
# build(InceptionResNetV2, 'InceptionResNetV2', 227)
# build(DenseNet121, 'DenseNet121', 227)
# build(Xception, 'Xception', 227)
# build(DenseNet201, 'DenseNet201', 227)
