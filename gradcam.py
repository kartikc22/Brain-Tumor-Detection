import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load trained model
model = load_model("best_model.keras")

# Class names
class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]
def preprocess_image(img_path):

    img = image.load_img(img_path, target_size=(224,224))

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img_array
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer_name).output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        pred_index = tf.argmax(predictions[0])

        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap,0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()
def save_and_display_gradcam(img_path, heatmap, alpha=0.4):

    img = cv2.imread(img_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img, 1-alpha, heatmap, alpha, 0)

    plt.figure(figsize=(8,8))

    plt.imshow(superimposed_img)

    plt.axis("off")

    plt.title("Grad-CAM Heatmap")

    plt.show()
if __name__ == "__main__":

    img_path = "sample_images/test1.jpg"

    img_array = preprocess_image(img_path)

    preds = model.predict(img_array)

    predicted_class = class_names[np.argmax(preds)]

    confidence = np.max(preds) * 100

    print("Prediction:", predicted_class)

    print("Confidence:", round(confidence,2), "%")

    last_conv_layer_name = "conv5_block3_out"

    heatmap = make_gradcam_heatmap(
        img_array,
        model,
        last_conv_layer_name
    )

    save_and_display_gradcam(img_path, heatmap)