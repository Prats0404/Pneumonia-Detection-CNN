# 🫁 Pneumonia Detection System using Deep Learning

A high-performance Deep Learning web application designed to detect Pneumonia from Chest X-Ray images. This project utilizes Convolutional Neural Networks (CNNs) and Transfer Learning to achieve high accuracy and is deployed using **Streamlit** for an interactive user interface.

## 🚀 Project Overview

Pneumonia is a life-threatening infectious disease affecting the lungs. Early detection is critical for effective treatment. This project automates the detection process by analyzing chest X-rays, classifying them into two categories:
* **Normal**
* **Pneumonia**

The system processes the input image, runs it through a trained CNN model, and provides a prediction with a confidence score.

## ✨ Key Features

* **Deep Learning Models:** Implements both a custom CNN built from scratch and a Transfer Learning model (MobileNetV2) for performance comparison.
* **Data Augmentation:** utilized techniques to balance the dataset and prevent overfitting.
* **Interactive Web App:** Built with Streamlit to allow users to upload images and get instant results.
* **Visualizations:** Includes confusion matrices and accuracy graphs (in the notebook) to analyze model performance.
* **Real-time Prediction:** Fast inference time using optimized model weights.

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **Web Framework:** Streamlit
* **Image Processing:** OpenCV, NumPy, Matplotlib
* **IDE:** VS Code, Jupyter Notebook

## 📂 Dataset

The model was trained on a dataset of Chest X-Ray images (Pneumonia vs. Normal).
* **Preprocessing:** Images were resized and normalized.
* **Augmentation:** Applied rotation, zoom, and horizontal flips to increase dataset diversity.

*(Note: The raw dataset is not included in this repo due to size constraints).*

## 📊 Model Architecture

We explored two approaches:
1.  **Custom CNN:** A sequential model with Conv2D, MaxPooling, and Dropout layers.
2.  **MobileNetV2 (Transfer Learning):** A pre-trained model fine-tuned for this specific binary classification task.

## 💻 How to Run locally

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Prats0404/Pneumonia-Detection-CNN.git](https://github.com/Prats0404/Pneumonia-Detection-CNN.git)
    cd Pneumonia-Detection-CNN
    ```

2.  **Install Dependencies**
    ```bash
    pip install tensorflow streamlit numpy opencv-python matplotlib
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

4.  **Usage**
    * The web app will open in your browser.
    * Upload a Chest X-Ray image (JPEG/PNG).
    * The model will display the result ("Normal" or "Pneumonia").

## 📸 Screenshots

*(You can add screenshots of your Streamlit app here later!)*

## 🤝 Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request.

---
*Developed by [Prats0404](https://github.com/Prats0404)*
