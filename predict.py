import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model("best_model.keras")

# Class labels (must match train_generator.class_indices)
class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

def predict_image(img_path):
    # Load image
    img = image.load_img(img_path, target_size=(224, 224))

    # Convert image to array
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)

    # Predicted class
    predicted_index = np.argmax(predictions)

    predicted_class = class_names[predicted_index]

    confidence = float(np.max(predictions) * 100)

    return predicted_class, confidence, predictions[0]
if __name__ == "__main__":

    img_path = "sample_images/test1.jpg"

    predicted_class, confidence, probabilities = predict_image(img_path)

    print("Prediction:", predicted_class)
    print("Confidence:", round(confidence, 2), "%")

    print("\nProbabilities:")

    for cls, prob in zip(class_names, probabilities):
        print(f"{cls}: {prob*100:.2f}%")