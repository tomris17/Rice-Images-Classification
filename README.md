# Rice Variety Classification with CNN & Streamlit Web App

This project builds a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images of 5 distinct rice species (Arborio, Basmati, Ipsala, Jasmine, and Karacadag) and deploys the trained model via a Streamlit web application.

The primary goal of this repository is to demonstrate how to train a high-accuracy deep learning model on a large image dataset (75,000 images) while keeping memory (RAM) usage low, and how to serve the model through an interactive web UI.

---

## Key Features

* **Memory-Efficient Data Loading:** Uses tf.data.Dataset pipelines to load and stream images without overloading system RAM.
* **Fast Training:** Optimized image resizing and batching for smooth execution.
* **High Performance:** Reaches ~99.2% accuracy on the validation set.
* **Visual Evaluation:** Includes training curve plots and a clear confusion matrix to analyze predictions.
* **Interactive Web Interface:** User-friendly Streamlit interface to upload rice images and retrieve real-time classification predictions with confidence scores.

---

## Performance & Results

* **Validation Accuracy:** ~99.19%
* **Classes:** Arborio, Basmati, Ipsala, Jasmine, Karacadag

---

## Tech Stack & Libraries

* **Python 3.x**
* **TensorFlow / Keras**
* **Streamlit**
* **NumPy**
* **Pillow (PIL)**
* **Matplotlib & Seaborn**
* **Scikit-learn**

---

## Dataset

The project uses the Rice Image Dataset consisting of 75,000 total images across 5 classes (15,000 images per class). 

* Data splits: 80% Training, 20% Validation
* Input image resolution: 128 x 128 x 3

---

## Web Application Setup

To run the Streamlit application locally:

1. Save your trained model as `rice_model.keras`.
2. Run the Streamlit app:

```bash
streamlit run app.py
