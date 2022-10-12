import os
from keras import optimizers
from keras.applications import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

"""Same architecture, and hyperparameters as previosuly developed model.
Run for less epochs as 800 takes too long.
Run on equalized images, not normal"""


BASE_DIR = '../labels'
TRAIN_DIR = os.path.join(BASE_DIR, 'train-equalized')
TEST_DIR = os.path.join(BASE_DIR, 'test-equalized')


BATCH_SIZE = 16

MODEL = InceptionV3(weights=None, 
  include_top=True, #Not adding own classifier
  input_shape=(227, 227, 3),
  classes=13)
MODEL.summary()

MODEL_NAME = 'InceptionV3 Scratch'
tbCallBack = TensorBoard(log_dir='../logs/InceptionV3_scratch/{}'.format(MODEL_NAME),
  histogram_freq=0,
  write_graph=True,
  write_images=True)

ADAM = optimizers.Adam(lr=0.001, beta_1=0.99, epsilon=0.001) 


DATAGEN_TRAIN = ImageDataGenerator(rescale=1. / 255,
	 rotation_range=180,
	 horizontal_flip=True,
	 zoom_range=0.2,
	 width_shift_range=0.2,
	 height_shift_range=0.2,
	 zca_whitening=True)

DATAGEN_TEST = ImageDataGenerator(rescale=1. / 255)

GENERATOR_TRAIN = DATAGEN_TRAIN.flow_from_directory(
      TRAIN_DIR,
      target_size=(227, 227),
      batch_size=BATCH_SIZE,
      class_mode='categorical',
      color_mode='rgb',  # useful to useful predct generator
      shuffle=True)

GENERATOR_TEST = DATAGEN_TEST.flow_from_directory(
      TRAIN_DIR,
      target_size=(227, 227),
      batch_size=BATCH_SIZE,
      class_mode='categorical', 
      color_mode='rgb', # useful to useful predct generator
      shuffle=True)

MODEL.compile(optimizer=ADAM,
              loss='categorical_crossentropy',
              metrics=['accuracy'])



TRAINING_STEPS_PER_EPOCH = len(GENERATOR_TRAIN.filenames) / BATCH_SIZE
TESTING_STEPS_PER_EPOCH = len(GENERATOR_TEST.filenames) / BATCH_SIZE

MODEL.fit_generator(GENERATOR_TRAIN,
          steps_per_epoch=TRAINING_STEPS_PER_EPOCH,
          epochs=800,  #Not going to do this many - too long.  Do 300
          validation_data=GENERATOR_TEST,
          validation_steps=TESTING_STEPS_PER_EPOCH,
          verbose=1,
          callbacks=[tbCallBack]) #Tensorboard stores history
