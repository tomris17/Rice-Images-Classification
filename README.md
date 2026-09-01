# Rice Variety Classification with Custom CNN, MobileNetV2 & Streamlit Web App

This repository features an end-to-end Deep Learning pipeline to classify 5 distinct rice species (Arborio, Basmati, Ipsala, Jasmine, and Karacadag). It evaluates a custom multi-layer CNN against a pre-trained **MobileNetV2 Transfer Learning** architecture and deploys the optimal model using an interactive Streamlit web application.

The core objective is to achieve high-precision classification on a large-scale image dataset (75,000 images) while maintaining memory-efficient data streaming and fast local CPU execution.

---

## Key Features

* **Dual Model Comparison:** Implements both a custom CNN (5 Conv2D + 3 Pooling + Dropout) and a pre-trained **MobileNetV2** Transfer Learning pipeline.
* **Memory-Efficient Data Pipeline:** Utilizes `tf.data.Dataset` streaming with `.cache()` and `.prefetch()` to prevent RAM overload during training.
* **High Performance:** Achieves **99.30% validation accuracy** with Transfer Learning.
* **Regularization & Metrics Tracking:** Incorporates Dropout layers to mitigate overfitting and tracks both Categorical Crossentropy Loss and Accuracy across training epochs.
* **Interactive Web UI:** User-friendly Streamlit interface to upload rice images and view real-time prediction probabilities and confidence scores.

---

## Performance & Results

| Model Architecture | Training Accuracy | Validation Accuracy | Validation Loss |
| :--- | :--- | :--- | :--- |
| **Custom CNN** | ~99.10% | 99.19% | ~0.0310 |
| **MobileNetV2 (Transfer Learning)** | **99.16%** | **99.30%** | **0.0229** |

* **Classes (5):** Arborio, Basmati, Ipsala, Jasmine, Karacadag

---

## Tech Stack & Libraries

* **Python 3.x**
* **TensorFlow / Keras**
* **Streamlit**
* **NumPy**
* **Pillow (PIL)**
* **Matplotlib & Seaborn**
* **Scikit-Learn**

---

## Dataset Overview

The project utilizes the **Rice Image Dataset** containing 75,000 total images across 5 classes (15,000 images per species).

* **Split Ratio:** 80% Training, 20% Validation
* **Target Resolution:** 128 x 128 x 3 pixels

---

## Web Application Setup

To run the Streamlit web application locally:

1. Ensure your trained model is saved in the root directory (e.g., `rice_mobilenet_model.keras` or `rice_model.h5`).
2. Launch the Streamlit server:

```bash
streamlit run app.py
