import os
import time
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, Input
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# Set environment variable to suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load and preprocess dataset using tf.data
def parse_image(filename, img_size=(200, 200)):
    parts = tf.strings.split(filename, os.sep)[-1]
    age = tf.strings.to_number(tf.strings.split(parts, "_")[0], out_type=tf.int32)
    gender = tf.strings.to_number(tf.strings.split(parts, "_")[1], out_type=tf.int32)
    img = tf.io.read_file(filename)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    img = img / 255.0  # Scale image
    return img, age, gender

def load_dataset(path, batch_size=32, img_size=(200, 200)):
    file_pattern = os.path.join(path, "*.jpg")
    dataset = tf.data.Dataset.list_files(file_pattern)
    dataset = dataset.map(lambda x: parse_image(x, img_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.cache()  # Cache the dataset if it can fit into memory
    dataset = dataset.repeat()  # Repeat the dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

# Timer for loading and preprocessing data
start_time = time.time()
print("Loading and preprocessing dataset...")

path = "UTKFace/UTKFace"
dataset = load_dataset(path)
print(f"Time taken to load and preprocess data: {time.time() - start_time} seconds")

# Split the data into training and testing sets
train_size = int(0.8 * 23708)
steps_per_epoch = train_size // 32  # Calculate steps per epoch

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

# Separate age and gender datasets
train_age_dataset = train_dataset.map(lambda img, age, gender: (img, age))
test_age_dataset = test_dataset.map(lambda img, age, gender: (img, age))
train_gender_dataset = train_dataset.map(lambda img, age, gender: (img, gender))
test_gender_dataset = test_dataset.map(lambda img, age, gender: (img, gender))

# Define and train the age model
print("Defining and training the age model...")

age_model = Sequential([
    Input(shape=(200, 200, 3)),
    Conv2D(32, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=3, strides=2),
    Conv2D(64, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=3, strides=2),
    Conv2D(128, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=3, strides=2),
    Flatten(),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dense(1, activation='linear', name='age')
])

age_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
age_model.summary()

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history_age = age_model.fit(
    train_age_dataset,
    validation_data=test_age_dataset,
    epochs=50,
    steps_per_epoch=steps_per_epoch,
    validation_steps=steps_per_epoch // 4,  # Assuming 20% of data for validation
    callbacks=[early_stopping]
)

age_model.save('age_model.h5')

# Define and train the gender model
print("Defining and training the gender model...")

gender_model = Sequential([
    Input(shape=(200, 200, 3)),
    Conv2D(32, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=3, strides=2),
    Conv2D(64, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=3, strides=2),
    Conv2D(128, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=3, strides=2),
    Flatten(),
    Dropout(0.2),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid', name='gender')
])

gender_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gender_model.summary()

history_gender = gender_model.fit(
    train_gender_dataset,
    validation_data=test_gender_dataset,
    epochs=50,
    steps_per_epoch=steps_per_epoch,
    validation_steps=steps_per_epoch // 4,
    callbacks=[early_stopping]
)

gender_model.save('gender_model.h5')
