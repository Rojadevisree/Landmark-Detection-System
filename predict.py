import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ====================================
# LOAD SAVED MODEL
# ====================================

model = tf.keras.models.load_model(
    "improved_landmark_model.keras"
)

print("Model loaded successfully!")

# ====================================
# SETTINGS
# ====================================

IMG_SIZE = (160, 160)

# IMPORTANT:
# Adjust labels according to your training
class_labels = {
    0: 'eiffel_tower',
    1: 'golden_temple',
    2: 'great_wall',
    3: 'hawa_mahal',
    4: 'qutub_minar',
    5: 'statue_of_liberty',
    6: 'taj_mahal'
}

# ====================================
# PREDICTION FUNCTION
# ====================================

def predict_image(img_path):

    img = tf.keras.preprocessing.image.load_img(
        img_path,
        target_size=IMG_SIZE
    )

    img_array = tf.keras.preprocessing.image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)[0]

    # Top 3 predictions
    top3 = np.argsort(pred)[-3:][::-1]

    print("\nTop Predictions:\n")

    for i in top3:
        print(f"{class_labels[i]}: {pred[i]*100:.2f}%")

    confidence = np.max(pred)

    print(
        f"\nFinal Prediction: "
        f"{class_labels[np.argmax(pred)]}"
    )

    # Show image
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# ====================================
# TEST IMAGE
# ====================================

predict_image("test.jpg")