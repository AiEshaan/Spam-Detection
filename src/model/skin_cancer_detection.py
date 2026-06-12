# The primary goal of this work is to build up a Model of Skin Cancer Detection System utilizing Machine Learning Algorithms. After experimenting with many different architectures for the CNN model It is found that adding the BatchNormalization layer after each Dense, and MaxPooling2D layer can help increase the validation accuracy. In future, a mobile application can be made.

# reference: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000/discussion/183083
# Data: https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000
# https://keras.io/api/models/sequential/
# https://keras.io/api/layers/core_layers/dense/
# https://keras.io/api/layers/merging_layers/add/
# https://keras.io/api/layers/convolution_layers/convolution2d
# https://keras.io/api/layers/convolution_layers/convolution2d
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization



import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import os
import numpy as np
from PIL import Image
import io
import base64

classes = {
    0: ("actinic keratoses and intraepithelial carcinomae(Cancer)"),
    1: ("basal cell carcinoma(Cancer)"),
    2: ("benign keratosis-like lesions(Non-Cancerous)"),
    3: ("dermatofibroma(Non-Cancerous)"),
    4: ("melanocytic nevi(Non-Cancerous)"),
    5: ("pyogenic granulomas and hemorrhage(Can lead to cancer)"),
    6: ("melanoma(Cancer)"),
}


model = Sequential()
model.add(
    Conv2D(
        16,
        kernel_size=(3, 3),
        input_shape=(28, 28, 3),
        activation="relu",
        padding="same",
        name="conv2d_0"
    )
)
model.add(MaxPool2D(pool_size=(2, 2), name="maxpool2d_0"))
model.add(BatchNormalization())
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv2d_1"))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2d_2"))
model.add(MaxPool2D(pool_size=(2, 2), name="maxpool2d_1"))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", name="conv2d_3"))
model.add(Conv2D(256, kernel_size=(3, 3), activation="relu", name="conv2d_4"))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(256, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(64, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(7, activation="softmax"))
model.summary()

# Load weights if file exists
# Get path to models folder relative to this file
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', '..', 'models')
model_path = os.path.join(models_dir, "best_model.h5")
if os.path.exists(model_path):
    model.load_weights(model_path)
else:
    print(f"Warning: {model_path} not found, using untrained weights for testing.")


def generate_gradcam(img_array, class_idx):
    # Create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer("conv2d_4").output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, class_idx]

    # This is the gradient of the output neuron (top predicted)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_gradcam_on_image(img_path, heatmap, alpha=0.4):
    # Load the original image
    img = Image.open(img_path).convert("RGB")
    img = img.resize((28, 28))
    img = np.array(img)

    # Resize heatmap to the same size as the image
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize the heatmap
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Resize the jet heatmap to original image size
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img


def gradcam_to_base64(superimposed_img):
    buffer = io.BytesIO()
    superimposed_img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
