# Fashion-MNIST Classification: Model Comparison

This project aims to compare the classification accuracy of different deep learning models in identifying clothing items from the Fashion-MNIST dataset. By evaluating a standard Dense Neural Network (DNN) and a Convolutional Neural Network (CNN), we identify which architecture better captures the spatial features of the image data.

## 📁 Project Structure

The project is organized into the following directory structure:

```
Fashion-MNIST/
├── SavedModel/
│   ├── UserDefineCNNModel.joblib
│   └── UserDefineModel.joblib
├── TrainingCode/
│   ├── UserDefineModel.py
│   └── UserDefineCNNModel.py
└── TrainingData/
    └── (Empty - Dataset is loaded via Keras API)
```

- **SavedModel/**: Contains the serialized, pre-trained models ready for deployment.
- **TrainingCode/**: Contains the Python scripts used for data preprocessing, model architecture definition, and training.
- **TrainingData/**: This directory is currently empty as the dataset is programmatically fetched using `keras.datasets`.

## 📊 Model Performance Comparison

We compared two distinct deep learning models based on their accuracy scores on the test dataset. The Convolutional Neural Network (CNN) serves as the primary performer.

| Model | Accuracy | Filename |
|-------|----------|----------|
| User Defined CNN Model | 90.40% | UserDefineCNNModel.joblib |
| User Defined Model (DNN) | 86.90% | UserDefineModel.joblib |

## 🛠️ Technical Details

### 1. Training Environment
- **Platform**: Python 3.x
- **Key Libraries**: TensorFlow, Keras, Scikit-Learn, Joblib

### 2. Dataset & Features
The models use the Fashion-MNIST dataset, which consists of 70,000 grayscale images in 10 categories.
- **Input (X)**: 28x28 pixel images of clothing items
- **Target Label (y)**: 10 Classes (T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

### 3. Model Architectures
- **User Defined Model (DNN)**: A simple architecture utilizing a `Flatten` layer followed by a `Dense` layer with 128 neurons (ReLU) and an output layer (Softmax).
- **User Defined CNN Model**: A more complex architecture featuring two `Conv2D` layers (32 and 64 filters), `MaxPool2D` layers, and a `Dense` layer with 256 neurons to capture spatial patterns.

### 4. Preprocessing Steps
- **Normalization**: Pixel values (0-255) were scaled to a range of 0 to 1 by dividing by 255.0.
- **Reshaping**: For the CNN model, input data was reshaped to `(28, 28, 1)` to include the single grayscale channel required by `Conv2D` layers.

## 💻 How to Load the Saved Models

You can use the following Python snippet to load the models from the `SavedModel` folder:

```python
import joblib

# Load the User Defined CNN Model
cnn_model = joblib.load('SavedModel/UserDefineCNNModel.joblib')

# Load the Standard User Defined Model
dnn_model = joblib.load('SavedModel/UserDefineModel.joblib')

# Make predictions (ensure data is reshaped appropriately for CNN)
# predictions = cnn_model.predict(X_test_reshaped)
```
