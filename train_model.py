# Landmark Detection 

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# CONFIG

IMG_SIZE = (160, 160)
BATCH_SIZE = 16
EPOCHS = 25
DATASET_PATH = "dataset"

# DATA GENERATORS

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

num_classes = train_generator.num_classes
class_labels = {v: k for k, v in train_generator.class_indices.items()}

# MODEL - TRANSFER LEARNING

base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(160, 160, 3)
)

# Freeze base model initially
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# CALLBACKS

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.3, patience=3)
]

# TRAINING (PHASE 1)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# FINE-TUNING (PHASE 2)

print("\nStarting Fine-Tuning...")

base_model.trainable = True

# Freeze first layers, train deeper layers
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    callbacks=callbacks
)

# EVALUATION

loss, acc = model.evaluate(val_generator)
print(f"\nFinal Accuracy: {acc * 100:.2f}%")

# PREDICTION FUNCTION

def predict_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0]

    # Top 3 predictions
    top3 = np.argsort(pred)[-3:][::-1]

    print("\nTop Predictions:")
    for i in top3:
        print(f"{class_labels[i]}: {pred[i]*100:.2f}%")

    confidence = np.max(pred)

    if confidence < 0.6:
        print("\nLow confidence → Unknown Landmark")
    else:
        print(f"\nFinal Prediction: {class_labels[np.argmax(pred)]} ({confidence*100:.2f}%)")

    plt.imshow(img)
    plt.axis('off')
    plt.show()

# TEST PREDICTION

predict_image("test.jpg")

# SAVE MODEL
model.save("improved_landmark_model.keras")
print("\nModel saved successfully!")