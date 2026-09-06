import argparse
import os

import numpy as np
import tensorflow as tf
from PIL import Image



# Agri AI - Tomato Leaf Disease Prediction
# Model: EfficientNetB0
# Input: 224 x 224
# Classes: 4

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Model",
    "AGRI_AI_Model.keras"
)

CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Spider Mites",
    "Target Spot"
]

IMAGE_SIZE = (224, 224)


def load_model():
    """Load the trained Keras model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.\n")

    return model


def prepare_image(image_path):
    """Load and prepare an image for the model."""

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(
            f"Could not open image: {e}"
        )

    # Resize to the same input size used during training
    image = image.resize(IMAGE_SIZE)

    # Convert image to NumPy array
    image_array = np.asarray(image, dtype=np.float32)

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict(model, image_path):
    """Run prediction and display results."""

    image = prepare_image(image_path)

    predictions = model.predict(image, verbose=0)

    # Convert model output into probabilities
    probabilities = predictions[0]

    # Find the class with the highest probability
    predicted_index = np.argmax(probabilities)

    predicted_class = CLASS_NAMES[predicted_index]
    confidence = probabilities[predicted_index] * 100

    print("=" * 45)
    print("           AGRI AI - PREDICTION")
    print("=" * 45)

    print(f"\nImage      : {image_path}")
    print(f"Prediction : {predicted_class}")
    print(f"Confidence : {confidence:.2f}%")

    print("\nProbabilities:")
    print("-" * 35)

    # Sort classes from highest to lowest probability
    sorted_indices = np.argsort(probabilities)[::-1]

    for index in sorted_indices:
        percentage = probabilities[index] * 100
        print(f"{CLASS_NAMES[index]:<18}: {percentage:6.2f}%")

    print("=" * 45)


def main():
    parser = argparse.ArgumentParser(
        description="Agri AI - Tomato Leaf Disease Classifier"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to the tomato leaf image"
    )

    args = parser.parse_args()

    try:
        model = load_model()
        predict(model, args.image)

    except FileNotFoundError as e:
        print(f"\nError: {e}")

    except ValueError as e:
        print(f"\nError: {e}")

    except Exception as e:
        print(f"\nUnexpected error: {e}")


if __name__ == "__main__":
    main()