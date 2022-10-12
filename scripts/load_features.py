import os
import numpy as np
from keras.callbacks import TensorBoard
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer



def train(train_dir):
    """Generates one hot labels for training data"""
    datagen_train = ImageDataGenerator(rescale=1. / 255)
    generator_train = datagen_train.flow_from_directory(train_dir,
        target_size=(150, 150),
        batch_size=1,
        class_mode='categorical',
        color_mode='rgb',  # useful to useful predct generator
        shuffle=False)
    labels_train = [filename.split('/', 1)[0] for filename in generator_train.filenames]
    print(labels_train)

    encoder = LabelBinarizer()
    encoder.fit(labels_train)
    labels_train_onehot = encoder.transform(labels_train)

    return labels_train_onehot

def test(test_dir):
    """Generates one hot labels for test data"""
    datagen_test = ImageDataGenerator(rescale=1. / 255)
    generator_test = datagen_test.flow_from_directory(test_dir,
        target_size=(150, 150),
        batch_size=1,
        class_mode='categorical',  # useful to useful predct generator
        shuffle=False)
    labels_test = [filename.split('/', 1)[0] for filename in generator_test.filenames]
    print(labels_test)

    encoder = LabelBinarizer()
    encoder.fit(labels_test)
    labels_test_onehot = encoder.transform(labels_test)

    return labels_test_onehot

def build(name):
    """Loads weights generated with export features, trains classifier"""
    tbCallBack = TensorBoard(log_dir='../logs/logs_different_conv_bases/{}'.format(name),
        histogram_freq=0, write_graph=True, write_images=True)

    base_dir = '../labels'
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')



    train_data = np.load(open('../different_model_weights/train_{}.npy'.format(name), 'rb'))
    train_labels = train(train_dir)

    validation_data = np.load(open('../different_model_weights/test_{}.npy'.format(name), 'rb'))
    validation_labels = test(test_dir)

    print("TRAIN DATA: {}".format(train_data.shape))

    Flatten()
    adm = optimizers.adam(lr=0.0001)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(13, activation='softmax'))
    model.compile(optimizer=adm,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    print(model.summary())

    print("Train data shape: {}\nTrain labels length: {}\nValidation data shape: {}\nValidation labels: {}".format(
        train_data.shape, len(train_labels), validation_data.shape, len(validation_labels)))

    model.fit(x=train_data,
        y=train_labels,
        epochs=50,
        batch_size=20,
        validation_data=(validation_data, validation_labels),
        callbacks=[tbCallBack])

build('VGG16')
