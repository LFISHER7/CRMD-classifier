import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers



BATCH_SIZE = 16

DATAGEN = ImageDataGenerator(
    rescale=1./255)  #Removed augmentation - same datagen for train/test

TRAIN_GENRATOR = DATAGEN.flow_from_directory(
    '../labels/train-binary',
    target_size=(227, 227),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True)

TEST_GENERATOR = DATAGEN.flow_from_directory(
    '../labels/test-binary',
    target_size=(227, 227),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True)

MODEL = models.Sequential()
MODEL.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(227, 227, 3)))
MODEL.add(layers.MaxPooling2D((2, 2)))
MODEL.add(layers.Conv2D(64, (3, 3), activation='relu'))
MODEL.add(layers.MaxPooling2D((2, 2)))
MODEL.add(layers.Conv2D(64, (3, 3), activation='relu'))
MODEL.add(layers.MaxPooling2D((2, 2)))
MODEL.add(layers.Conv2D(64, (3, 3), activation='relu'))
MODEL.add(layers.MaxPooling2D((2, 2)))
MODEL.add(layers.Flatten())
MODEL.add(layers.Dense(64, activation='relu'))
MODEL.add(layers.Dense(2, activation='sigmoid')) # Change from softmax to sigmoid for binary
OPT = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
MODEL.compile(loss="binary_crossentropy", optimizer=OPT, metrics=['accuracy']) #Changed to binary

TRAINING_STEPS_PER_EPOCH = len(TRAIN_GENRATOR.filenames) /  BATCH_SIZE
TESTING_STEPS_PER_EPOCH = len(TEST_GENERATOR.filenames) /  BATCH_SIZE

HISTORY = MODEL.fit_generator(
    TRAIN_GENRATOR,
    steps_per_epoch=TRAINING_STEPS_PER_EPOCH,
    epochs=30,
    validation_data=TEST_GENERATOR,
    validation_steps=TESTING_STEPS_PER_EPOCH,
    verbose=1)


ACC = HISTORY.history['acc']
TEST_ACC = HISTORY.history['val_acc']
LOSS = HISTORY.history['loss']
TEST_LOSS = HISTORY.history['val_loss']

EPOCHS = range(1, len(ACC) + 1)

plt.plot(EPOCHS, ACC, 'bo', label='Training acc')
plt.plot(EPOCHS, TEST_ACC, 'b', label='Validation acc')
plt.title('Training and test accuracy')
plt.legend()

plt.figure()

plt.plot(EPOCHS, LOSS, 'bo', label='Training loss')
plt.plot(EPOCHS, TEST_LOSS, 'b', label='Test_loss')
plt.title('Training and test loss')
plt.legend()

plt.show()