import streamlit as st
import numpy as np
import tensorflow as tf
import cv2

from tensorflow.keras.models import load_model
from PIL import Image


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0px 3px 12px rgba(0,0,0,0.08);
    margin-top: 15px;
}

.prediction {
    font-size: 30px;
    font-weight: bold;
}

.confidence {
    font-size: 22px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="title">🧠 Brain Tumor Detection AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Deep Learning Based Brain MRI Classification using ResNet50'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# MODEL
# =========================================================

@st.cache_resource
def load_brain_model():
    return load_model("best_model.keras")


try:
    model = load_brain_model()
except Exception as e:
    st.error("❌ Could not load the trained model.")
    st.exception(e)
    st.stop()


# =========================================================
# CLASSES
# =========================================================

class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]


# =========================================================
# DISEASE INFORMATION
# =========================================================

disease_info = {
    "Glioma": {
        "title": "🧠 Glioma",
        "description": """
Glioma is a tumor that develops from glial cells in the
brain or spinal cord.

Its behavior can vary depending on its type, grade and
location. Possible symptoms include headaches, seizures,
vision or speech changes, memory problems and behavioral
changes.
"""
    },

    "Meningioma": {
        "title": "🧠 Meningioma",
        "description": """
Meningioma is a tumor that develops from the meninges,
the protective membranes surrounding the brain and spinal
cord.

Many meningiomas grow slowly. Depending on their size and
location, they may cause headaches, seizures, vision
problems, weakness or cognitive changes.
"""
    },

    "No Tumor": {
        "title": "✅ No Tumor Detected",
        "description": """
The model classified this MRI image into the "No Tumor"
category.

This category represents MRI images labeled as not
containing a brain tumor in the dataset used for training.

This prediction should not be considered medical
confirmation that a person is free from disease.
"""
    },

    "Pituitary": {
        "title": "🧠 Pituitary Tumor",
        "description": """
Pituitary tumors develop in the pituitary gland, located
at the base of the brain.

Many are non-cancerous. Depending on their size and effect
on hormone production, they may cause headaches, vision
changes or hormonal changes.
"""
    }
}


# =========================================================
# PREPROCESS IMAGE
# =========================================================

def preprocess_image(uploaded_file):
    """Load and prepare MRI image for the model."""

    img = Image.open(uploaded_file).convert("RGB")
    resized = img.resize((224, 224))

    img_array = np.array(resized).astype("float32")
    img_array /= 255.0

    img_array = np.expand_dims(img_array, axis=0)

    return img, img_array


# =========================================================
# GRAD-CAM
# =========================================================

def make_gradcam(img_array, predicted_index):
    """Generate Grad-CAM heatmap."""

    last_conv_layer = model.get_layer("conv5_block3_out")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv_layer.output,
            model.output
        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        class_channel = predictions[:, predicted_index]

    gradients = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    max_value = tf.reduce_max(heatmap)

    if max_value > 0:
        heatmap /= max_value

    return heatmap.numpy()


# =========================================================
# CREATE GRAD-CAM OVERLAY
# =========================================================

def create_gradcam_overlay(original_image, heatmap):
    """Overlay Grad-CAM heatmap on original MRI."""

    original = np.array(original_image)

    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    return overlay


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🧠 About This Project")

    st.write(
        "A deep learning application that classifies "
        "brain MRI images into four categories."
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write("Architecture: ResNet50")
    st.write("Input Size: 224 × 224")
    st.write("Classes: 4")
    st.write("Framework: TensorFlow / Keras")

    st.divider()

    st.subheader("📂 Classes")

    for cls in class_names:
        st.write(f"• {cls}")

    st.divider()

    st.subheader("📈 Performance")

    st.metric(
        "Test Accuracy",
        "74.87%"
    )

    st.divider()

    st.info(
        "Educational and research project only. "
        "The prediction is not a medical diagnosis."
    )


# =========================================================
# UPLOAD
# =========================================================

st.header("📤 Upload Brain MRI")

st.write(
    "Upload an MRI image to classify it using the "
    "trained ResNet50 model."
)

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# MAIN APPLICATION
# =========================================================

if uploaded_file is None:

    st.info(
        "👆 Upload a brain MRI image above to start "
        "the prediction."
    )

else:

    # -----------------------------------------------------
    # IMAGE PREPROCESSING
    # -----------------------------------------------------

    original_image, img_array = preprocess_image(
        uploaded_file
    )

    st.divider()

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    predictions = model.predict(
        img_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index] * 100
    )

    # -----------------------------------------------------
    # IMAGE + RESULT
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🖼️ Uploaded MRI")

        st.image(
            original_image,
            use_container_width=True
        )

    with col2:

        st.subheader("🔍 AI Prediction")

        st.markdown(
            f"""
            <div class="result-box">
                <div class="prediction">
                    {predicted_class}
                </div>
                <br>
                <div class="confidence">
                    Model Confidence: {confidence:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # PROBABILITIES
    # -----------------------------------------------------

    st.divider()

    st.subheader("📊 Class Probabilities")

    for cls, probability in zip(
        class_names,
        predictions
    ):

        percentage = float(
            probability * 100
        )

        st.write(
            f"**{cls}: {percentage:.2f}%**"
        )

        st.progress(
            float(probability)
        )

    # -----------------------------------------------------
    # DISEASE INFORMATION
    # -----------------------------------------------------

    st.divider()

    st.header("📚 About the Predicted Condition")

    info = disease_info[predicted_class]

    st.subheader(info["title"])

    st.write(info["description"])

    st.warning(
        """
        ⚠️ **Medical Disclaimer**

        This AI prediction is for educational and research
        purposes only. It is not a medical diagnosis and
        should not replace evaluation by a qualified
        healthcare professional.
        """
    )

   

    # -----------------------------------------------------
    # GRAD-CAM
    # -----------------------------------------------------

    st.divider()

    st.header("🔥 Grad-CAM Visualization")

    st.write(
        """
        Grad-CAM (Gradient-weighted Class Activation Mapping)
        helps visualize the regions of the MRI that contributed
        to the model's prediction.
        """
    )

    try:

        heatmap = make_gradcam(
            img_array,
            predicted_index
        )

        gradcam_image = create_gradcam_overlay(
            original_image,
            heatmap
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("🖼️ Original MRI")

            st.image(
                original_image,
                use_container_width=True
            )

        with col2:

            st.subheader("🔥 Grad-CAM Heatmap")

            st.image(
                gradcam_image,
                use_container_width=True
            )

        st.success(
            "✅ Grad-CAM generated successfully."
        )

    except Exception as e:

        st.error(
            "❌ Grad-CAM could not be generated."
        )

        st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🧠 Brain Tumor Detection | ResNet50 | "
    "TensorFlow/Keras | Grad-CAM | "
    "Educational Project"
)