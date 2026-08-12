import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.cm as cm

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_trained_model():
    return load_model("best_model.keras")

model = load_trained_model()

# Class Names
class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# -------------------------------
# Prediction Function
# -------------------------------
def predict(img):

    img = img.resize((224,224))

    img_array = image.img_to_array(img)

    img_array = img_array/255.0

    img_array = np.expand_dims(img_array,axis=0)

    prediction = model.predict(img_array, verbose=0)

    predicted_index = np.argmax(prediction)

    confidence = float(np.max(prediction)*100)

    return prediction, predicted_index, confidence


# -------------------------------
# GradCAM
# -------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output,
         model.output]
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


# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Detection using ResNet50")

st.write("Upload an MRI Scan to detect Brain Tumor.")

uploaded_file = st.file_uploader(
    "Choose MRI Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    col1,col2 = st.columns(2)

    with col1:
        st.image(img,width=350)

    if st.button("Predict"):

        prediction,predicted_index,confidence = predict(img)

        with col2:

            st.subheader("Prediction")

            st.success(class_names[predicted_index])

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.subheader("Class Probabilities")

            for i,name in enumerate(class_names):

                st.progress(float(prediction[0][i]))

                st.write(
                    f"{name} : {prediction[0][i]*100:.2f}%"
                )

        # -------------------------------
        # GradCAM
        # -------------------------------

        img2 = img.resize((224,224))

        img_array = image.img_to_array(img2)

        img_array = img_array/255.0

        img_array = np.expand_dims(img_array,axis=0)

        heatmap = make_gradcam_heatmap(
            img_array,
            model,
            "conv5_block3_out"
        )

        heatmap = np.uint8(255*heatmap)

        jet = cm.get_cmap("jet")

        jet_colors = jet(np.arange(256))[:,:3]

        jet_heatmap = jet_colors[heatmap]

        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)

        jet_heatmap = jet_heatmap.resize((224,224))

        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

        original = np.array(img2)

        superimposed = jet_heatmap*0.4 + original

        superimposed = np.uint8(superimposed)

        st.subheader("Grad-CAM Heatmap")

        st.image(superimposed,width=350)