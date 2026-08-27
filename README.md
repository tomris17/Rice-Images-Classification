# Rice Variety Classification with CNN

This project builds a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify images of 5 distinct rice species (Arborio, Basmati, Ipsala, Jasmine, and Karacadag).

The primary goal of this repository is to demonstrate how to train a high-accuracy deep learning model on a large image dataset (75,000 images) while keeping memory (RAM) usage low and training efficient.

---

## Key Features

* **Memory-Efficient Data Loading:** Uses tf.data.Dataset pipelines to load and stream images without overloading system RAM.
* **Fast Training:** Optimized image resizing and batching for smooth CPU execution.
* **High Performance:** Reaches ~99.2% accuracy on the validation set.
* **Visual Evaluation:** Includes training curve plots and a clear confusion matrix to analyze predictions.

---

## Performance & Results

* **Validation Accuracy:** ~99.19%
* **Classes:** Arborio, Basmati, Ipsala, Jasmine, Karacadag

---

## Tech Stack & Libraries

* **Python 3.x**
* **TensorFlow / Keras**
* **NumPy**
* **Matplotlib & Seaborn**
* **Scikit-learn**

---

## Dataset

The project uses the Rice Image Dataset consisting of 75,000 total images across 5 classes (15,000 images per class). 

* Data splits: 80% Training, 20% Validation
* Input image resolution: 128 x 128 x 3
