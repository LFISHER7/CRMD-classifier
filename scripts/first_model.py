import matplotlib.pyplot as plt
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator


BATCH_SIZE = 16  # Can change this hyperparameter

TRAIN_DATAGEN = ImageDataGenerator(
    rescale=1./255, #Puts all pixel values in range0-1
    rotation_range=90,
    horizontal_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='constant') #How it fills newly created pixels after shifting etc

TEST_DATAGEN = ImageDataGenerator(rescale=1./255) # Dont augment test set

TRAIN_GENERATOR = TRAIN_DATAGEN.flow_from_directory(
	'../labels/train',
	target_size=(227, 227),
	batch_size=BATCH_SIZE,
	class_mode='categorical')

TEST_GENERATOR = TEST_DATAGEN.flow_from_directory(
	'../labels/test',
	target_size=(227, 227),
	batch_size=BATCH_SIZE,
	class_mode='categorical') # use categorical rather than binary

MODEL = models.Sequential()  # Sequential model is linear stack

MODEL.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu',
	input_shape=(227, 227, 3)))
MODEL.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
MODEL.add(layers.MaxPooling2D(pool_size=(2, 2)))
MODEL.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
MODEL.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
MODEL.add(layers.MaxPooling2D(pool_size=(2, 2)))
MODEL.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
MODEL.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
MODEL.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
MODEL.add(layers.MaxPooling2D(pool_size=(2, 2)))
MODEL.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
MODEL.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
MODEL.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
MODEL.add(layers.MaxPooling2D(pool_size=(2, 2)))
MODEL.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
MODEL.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
MODEL.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
MODEL.add(layers.MaxPooling2D(pool_size=(2, 2)))
MODEL.add(layers.Flatten())
MODEL.add(layers.Dense(256, activation='relu'))
MODEL.add(layers.Dropout(0.58))
MODEL.add(layers.Dense(13, activation='softmax'))  #13 outputs as 13 classes.  Softmax is probability

MODEL.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

TRAINING_STEPS_PER_EPOCH = len(TRAIN_GENERATOR.filenames) / BATCH_SIZE # So get 800 images
TESTING_STEPS_PER_EPOCH = len(TEST_GENERATOR.filenames) / BATCH_SIZE


HISTORY = MODEL.fit_generator(
	TRAIN_GENERATOR,
	steps_per_epoch=TRAINING_STEPS_PER_EPOCH,
	epochs=30,
	validation_data=TEST_GENERATOR,
	validation_steps=TESTING_STEPS_PER_EPOCH,
	verbose=1) # verbose shows you results as it runs

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
