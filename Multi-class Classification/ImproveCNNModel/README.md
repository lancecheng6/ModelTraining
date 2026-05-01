# ImproveCNNModel: Flower Classification & Data Augmentation

This project compares the classification performance of Convolutional Neural Networks (CNN) using the Flower Photos dataset. The primary focus is to evaluate how Data Augmentation techniques—such as random rotations, shifts, and brightness adjustments—improve the model's accuracy and ability to generalize compared to a standard image generator.

## 📁 Project Structure

The project is organized into the following directory structure:

```
ImproveCNNModel/
├── SavedModel/
│   ├── CNNModelWithImageDataGenerator.joblib
│   └── CNNModelWithImageDataGeneratorAndDataArgmentation.joblib
├── TrainingCode/
│   ├── CNNModelWithImageDataGenerator.py
│   └── CNNModelWithImageDataGeneratorAndDataArgmentation.py
└── README.md
```

- **SavedModel/**: Contains serialized, pre-trained models stored using the joblib library.
- **TrainingCode/**: Contains the Python scripts for data preprocessing, CNN architecture definition, and the training pipeline.

## 📊 Model Performance Comparison

The following table shows the final validation accuracy for both models. The results demonstrate that incorporating data augmentation significantly improved the model's performance on the dataset.

| Model | Accuracy | Preprocessing Strategy | Filename |
|-------|----------|------------------------|----------|
| Standard CNN | 53.16% | Basic pixel normalization (rescaling) | CNNModelWithImageDataGenerator.joblib |
| Augmented CNN | 65.57% | Normalization + Random rotation, zoom, shift, and flip | CNNModelWithImageDataGeneratorAndDataArgmentation.joblib |

## 🛠️ Technical Details

### 1. Training Environment
- **Language**: Python 3.x
- **Core Libraries**: TensorFlow 2.x, Keras, Scikit-Learn, Joblib, Matplotlib

### 2. Dataset & Classes
The models utilize the Google flower_photos dataset, which includes five categories:
- **Target Classes**: Daisy, Dandelion, Roses, Sunflowers, and Tulips.
- **Input Size**: RGB images resized to $192 \times 192$ pixels.

### 3. Model Architecture
Both models implement a sequential CNN architecture:
- **Conv2D Layer 1**: 32 filters, $5 \times 5$ kernel, ReLU activation.
- **MaxPooling Layer 1**: $2 \times 2$ pool size.
- **Conv2D Layer 2**: 64 filters, $3 \times 3$ kernel.
- **MaxPooling Layer 2**: $2 \times 2$ pool size.
- **Dense Layer**: 128 neurons with ReLU activation.
- **Output Layer**: 5 neurons with Softmax activation.

### 4. Data Augmentation Parameters
The augmented model uses ImageDataGenerator to apply real-time transformations during training:
- **Rotation Range**: 40 degrees
- **Width/Height Shift**: 20% of image dimensions
- **Shear/Zoom**: 20% range
- **Brightness Range**: 0.6 to 1.2
- **Horizontal Flip**: Enabled

## 💻 How to Load the Saved Models

You can load the pre-trained models from the `SavedModel/` folder using the following Python snippet:

```python
import joblib

# Load the Augmented CNN Model
model = joblib.load('SavedModel/CNNModelWithImageDataGeneratorAndDataArgmentation.joblib')

# Ensure your input data is normalized (1./255) and resized to (192, 192, 3)
# prediction = model.predict(preprocessed_image_batch)
```
