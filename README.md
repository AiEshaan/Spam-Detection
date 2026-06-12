# Skin Cancer Detection

A deep learning-based skin cancer classification system using Convolutional Neural Networks (CNN) built with Keras and TensorFlow. This project includes a Flask web application for easy interaction.

## Features

- **Multi-Image Upload**: Upload multiple skin lesion images at once for batch analysis
- **Confidence Score Display**: Visual confidence scores for all prediction classes using progress bars
- **Prediction History**: Local browser storage to track and view past predictions
- **Modern UI/UX**: Clean, responsive design with smooth interactions
- **Detailed Condition Info**: Information about each skin condition, including potential causes and symptoms
- **Disclaimer**: Prominent medical disclaimer to remind users this is not a substitute for professional care

## Installation

### Prerequisites

- Python 3.7+
- pip (Python package manager)

### Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/AiEshaan/Spam-Detection.git
   cd Skin-Cancer-Detection-MNIST
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model (optional but recommended for optimal performance)**
   First, download the dataset from [Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000), then run:
   ```bash
   python train_model.py
   ```
   This will generate `best_model.h5`

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your browser and navigate to `http://localhost:5000`

## Usage

1. Go to the home page
2. Click on "Choose File(s)" to upload one or more skin lesion images
3. Click "Analyze Image(s)"
4. View the prediction results with confidence scores
5. Check your prediction history on the home page

## Project Structure

```
Skin-Cancer-Detection-MNIST/
├── app.py                      # Flask web application
├── skin_cancer_detection.py    # CNN model definition
├── train_model.py              # Script to train the model
├── requirements.txt            # Project dependencies
├── templates/                  # HTML templates
│   ├── home.html              # Home page
│   ├── reults.html            # Single image result page
│   └── multi_results.html     # Multi-image results page
├── Skin_Cancer_Detection.ipynb
├── model_architecture.png
├── model.png
└── README.md
```

## Model Architecture

The model is a deep CNN with the following architecture:

- **Input**: 28x28x3 RGB images
- **Convolutional Layers**: Multiple Conv2D layers for feature extraction
- **Pooling Layers**: MaxPooling2D layers for downsampling
- **Batch Normalization**: Applied after Dense and MaxPooling2D layers to stabilize training
- **Dropout Layers**: Regularization to prevent overfitting
- **Dense Layers**: Classification head with softmax activation

Total parameters: ~504k

## Classes Detected

The model classifies skin lesions into 7 categories:

1. Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)
2. Basal cell carcinoma (bcc)
3. Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses) (bkl)
4. Dermatofibroma (df)
5. Melanoma (mel)
6. Melanocytic nevi (nv)
7. Vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage) (vasc)

## Dataset

This project uses the [HAM10000 dataset](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000) from Kaggle, which contains 10,015 dermatoscopic images of skin lesions.

## License

This project is licensed under the MIT License.

## Disclaimer

**IMPORTANT**: This tool is for educational purposes only and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified dermatologist or healthcare provider for any concerns about skin lesions or skin cancer.
