# Human Action Recognition with CNN + LSTM

![Project Banner](https://img.shields.io/badge/Deep%20Learning-CNN%20%2B%20LSTM-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Flask](https://img.shields.io/badge/Backend-Flask-lightgrey)

A complete Deep Learning application for Human Action Recognition. This project uses a hybrid **CNN (MobileNetV2)** and **LSTM** architecture to classify human actions from images. It features a Flask backend and a modern, responsive web frontend for real-time predictions.

## 🚀 Features

-   **Deep Learning Model**: Combines Convolutional Neural Networks (CNN) for spatial feature extraction and Long Short-Term Memory (LSTM) for sequence/context reasoning.
-   **User-Friendly Interface**: Modern, responsive web UI with drag-and-drop image upload.
-   **Real-Time Prediction**: Instant analysis with confidence scores and detailed class probabilities.
-   **REST API**: Flask-based backend exposing a `/predict` endpoint.

## 📂 Project Structure

```
AssignmentDeepLearning/
├── app/
│   ├── app.py              # Flask backend application
│   └── models/             # Trained models and class indices
│       ├── action_recognition_model.h5
│       └── class_indices.json
├── DataSet/                # Dataset directory (Excluded from Git)
├── scripts/
│   └── train_model.py      # Script to train the CNN+LSTM model
├── static/
│   ├── css/                # Frontend styling
│   └── js/                 # Frontend logic
├── templates/
│   └── index.html          # Main HTML interface
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🛠️ Installation & Setup

### Prerequisites

-   Python 3.8 or higher
-   pip (Python package manager)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AssignmentDeepLearning.git
cd AssignmentDeepLearning
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset (Optional for Running, Required for Training)

If you wish to retrain the model, place your dataset in the `DataSet/` directory. The structure should be:

```
DataSet/
└── train/
    ├── Image_1.jpg
    ├── Image_2.jpg
    ...
└── train.csv  # Contains filename and label columns
```

### 4. Run the Application

Start the Flask server:

```bash
python app/app.py
```

Open your browser and navigate to: **http://localhost:5000**

## 🧠 Model Architecture

The model uses a hybrid approach:

1.  **Spatial Features (CNN)**: We use **MobileNetV2** (pre-trained on ImageNet) to extract high-level features from the input image (224x224x3).
2.  **Feature Reshaping**: The output feature map (7x7x1280) is reshaped into a sequence of 49 vectors of size 1280.
3.  **Reasoning (LSTM)**: An **LSTM layer** (128 units) processes this sequence to capture "contextual" dependencies within the image structure.
4.  **Classification**: A Dense layer with Softmax activation predicts the action class.

## 📸 Screenshots


<img width="975" height="482" alt="image" src="https://github.com/user-attachments/assets/5876ebff-9ced-40b0-8400-b8aa7e26610d" />
<img width="975" height="586" alt="image" src="https://github.com/user-attachments/assets/73e21953-fe45-446d-af38-307e9e52fe8f" />
<img width="975" height="556" alt="image" src="https://github.com/user-attachments/assets/8e3bfbff-3274-47f7-9137-471340bb871f" />
<img width="975" height="612" alt="image" src="https://github.com/user-attachments/assets/5645a89a-8f74-40b0-8ead-17b99d235296" />
<img width="975" height="485" alt="image" src="https://github.com/user-attachments/assets/e0e2fce7-5bee-4245-8dc0-e0a0da882a91" />
<img width="975" height="596" alt="image" src="https://github.com/user-attachments/assets/c98c80ee-b4e8-4cb9-95d7-3491507984ce" />
<img width="975" height="562" alt="image" src="https://github.com/user-attachments/assets/15b69d0e-9992-42a7-b97b-f2646b2c18e9" />
<img width="975" height="554" alt="image" src="https://github.com/user-attachments/assets/eb0d2c52-8cc3-4f18-8a2b-1fb1ac443032" />
<img width="975" height="568" alt="image" src="https://github.com/user-attachments/assets/6cf0054d-0687-4023-aedf-fc788c8e16e8" />


## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## 📄 License

This project is open-source and available under the MIT License.
