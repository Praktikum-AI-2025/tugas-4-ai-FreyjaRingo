# =====================================================================================================
# Membangun sebuah model Neural Network untuk klasifikasi dataset Horse or Human dalam binary classes.
#
# Input layer harus menerima 150x150 dengan 3 bytes warna sebagai input shapenya.
# Jangan menggunakan lambda layers dalam model.
#
# Dataset yang digunakan dibuat oleh Laurence Moroney (laurencemoroney.com).
#
# Standar yang harus dicapai untuk accuracy dan validation_accuracy > 83%
# =====================================================================================================

import urllib.request
import zipfile
import tensorflow as tf
import os
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

data_url_1 = 'https://github.com/dicodingacademy/assets/releases/download/release-horse-or-human/horse-or-human.zip'
urllib.request.urlretrieve(data_url_1, 'horse-or-human.zip')
local_file = 'horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_file, 'r')
zip_ref.extractall('data/horse-or-human')

data_url_2 = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/validation-horse-or-human.zip'
urllib.request.urlretrieve(data_url_2, 'validation-horse-or-human.zip')
local_file = 'validation-horse-or-human.zip'
zip_ref = zipfile.ZipFile(local_file, 'r')
zip_ref.extractall('data/validation-horse-or-human')
zip_ref.close()

def solution_05():
    TRAINING_DIR = 'data/horse-or-human'
    VALIDATION_DIR = 'data/validation-horse-or-human'
    
    train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Mempersiapkan generator untuk training data
    train_generator = train_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )
    
    # Mempersiapkan generator untuk validation data
    validation_generator = validation_datagen.flow_from_directory(
        VALIDATION_DIR,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary'
    )

    # Membangun model CNN
    model = tf.keras.models.Sequential([
        # Input layer
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Layer kedua
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Layer ketiga
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Flatten layer
        tf.keras.layers.Flatten(),
        
        # Dense layers
        tf.keras.layers.Dense(128, activation='relu'), 
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(learning_rate=0.0001),
        metrics=['accuracy']
    )
    
    return model

model = solution_05()

TRAINING_DIR = 'data/horse-or-human'
VALIDATION_DIR = 'data/validation-horse-or-human'

# Data augmentation untuk training
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    horizontal_flip=True,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode='nearest'
)

# Hanya rescaling untuk validasi
validation_datagen = ImageDataGenerator(rescale=1./255)

# Generator untuk training data
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Generator untuk validation data
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True
)

# Melatih model
history = model.fit(
    train_generator,
    steps_per_epoch=25,  # ~800 images / 32 batch size
    epochs=15,
    validation_data=validation_generator,
    validation_steps=8,  # ~256 validation images / 32 batch size
    verbose=1,
    callbacks=[early_stopping]
)

# Hasil final dari proses training
print("\n=== HASIL FINAL DARI PROSES TRAINING ===")
print(f"Training Accuracy (Epoch Terakhir): {history.history['accuracy'][-1]:.4f}")
print(f"Validation Accuracy (Epoch Terakhir): {history.history['val_accuracy'][-1]:.4f}")

# Evaluasi model pada data validasi
print("\n=== HASIL EVALUASI PADA DATA VALIDASI ===")
evaluation = model.evaluate(validation_generator, verbose=1)
print(f"Evaluation Loss: {evaluation[0]:.4f}")
print(f"Evaluation Accuracy: {evaluation[1]:.4f}")

# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_05()
    model.save("model_05.h5")
