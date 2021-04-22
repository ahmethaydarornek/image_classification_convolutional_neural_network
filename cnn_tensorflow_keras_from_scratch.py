# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 18:06:37 2021

@author: ahmethaydarornek

This script is created to train a convolutional neural network from scratch,
the created model can be extended by adding more convolution, pooling
and dense layers.
"""

import tensorflow

# Creating an emptpy model to fill with convolution, pooling and dense layers.
model = tensorflow.keras.models.Sequential()

model.add(tensorflow.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(tensorflow.keras.layers.MaxPooling2D((2, 2)))
model.add(tensorflow.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tensorflow.keras.layers.MaxPooling2D((2, 2)))
model.add(tensorflow.keras.layers.Conv2D(64, (3, 3), activation='relu'))

# 2D outputs are converted to a vector.
model.add(tensorflow.keras.layers.Flatten())
model.add(tensorflow.keras.layers.Dense(512, activation='relu'))
# Dropout is used to prevent the overfitting problem.
model.add(tensorflow.keras.layers.Dropout(0.3))
model.add(tensorflow.keras.layers.Dense(128, activation='relu'))
model.add(tensorflow.keras.layers.Dropout(0.3))
model.add(tensorflow.keras.layers.Dense(2, activation='softmax'))

model.compile(loss='binary_crossentropy',
              optimizer=tensorflow.keras.optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])

# Showing the filled model.
model.summary()

# Defining the directories that data are in.
train_dir = 'data/train'
validation_dir = 'data/val'
test_dir = 'data/test'

# We need to apply data augmentation methods to prevent overfitting.
train_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
      rescale=1./255, # piksel değerleri 0-255'den 0-1 arasına getiriliyor.
      rotation_range=40, # istenilen artırma işlemleri yapılabilir.
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest'
      )

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=20,
        )

# To validate the training process, we do not need augmented images.
validation_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
        )

validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=20,
        )

# Training the model.
history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=50,
      validation_data=validation_generator,
      validation_steps=5)

# Saving the trained model to working directory.
model.save('trained_tf_model.h5')

# To test the trained model, we do not need augmented images.
test_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
        )

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=20,
        )

# Printing the test results.
test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print('test acc:', test_acc)

