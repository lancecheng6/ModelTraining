# ImproveCNNModel: Flower Classification & Data Augmentation

This project compares the classification performance of Convolutional Neural Networks (CNN) using the Flower Photos dataset. The primary focus is to evaluate how Data Augmentation techniques, Transfer Learning (using VGG16), and Fine-Tuning improve the model's accuracy and ability to generalize.

## 📁 Project Structure

The project is organized into the following directory structure:

```
ImproveCNNModel/
├── SavedModel/
│   ├── CNNModelWithImageDataGenerator.joblib
│   ├── CNNModelWithImageDataGeneratorAndDataArgmentation.joblib
│   ├── TransferLearningUsingVGG16AndUserDefineClassifier.joblib
│   └── FineTuningVGG16AndUserDefineClassifier.joblib
├── TrainingCode/
│   ├── CNNModelWithImageDataGenerator.py
│   ├── CNNModelWithImageDataGeneratorAndDataArgmentation.py
│   ├── TransferLearningUsingVGG16AndUserDefineClassifier.py
│   └── FineTuningVGG16AndUserDefineClassifier.py
└── README.md
```

- **SavedModel/**: Contains serialized, pre-trained models stored using the joblib library.
- **TrainingCode/**: Contains the Python scripts for data preprocessing, CNN architecture definition, and the training pipeline.

## 📊 Model Performance Comparison

The following table shows the final validation accuracy for all models. The results demonstrate that using pre-trained models like VGG16, especially with fine-tuning, significantly outperforms standard CNN architectures.

| Model | Accuracy | Preprocessing / Strategy | Filename |
|-------|----------|---------------------------|----------|
| Standard CNN | 53.16% | Basic pixel normalization | CNNModelWithImageDataGenerator.joblib |
| Augmented CNN | 65.57% | Normalization + Data Augmentation | CNNModelWithImageDataGeneratorAndDataArgmentation.joblib |
| VGG16 Transfer Learning | 82.2% | Frozen VGG16 Base + Custom Classifier | TransferLearningUsingVGG16AndUserDefineClassifier.joblib |
| VGG16 Fine-Tuning | 91.6% | Partial Unfreeze VGG16 + Custom Classifier | FineTuningVGG16AndUserDefineClassifier.joblib |

## 🛠️ Technical Details

### 1. Training Environment
- **Language**: Python 3.x
- **Core Libraries**: TensorFlow 2.x, Keras, Scikit-Learn, Joblib, Matplotlib

### 2. Dataset & Classes
The models utilize the Google flower_photos dataset:
- **Target Classes**: Daisy, Dandelion, Roses, Sunflowers, and Tulips.
- **Input Size**: RGB images resized to $192 \times 192$ pixels.

### 3. Model Architectures
#### Standard CNN (Sequential)
- **Conv2D Layer 1**: 32 filters, $5 \times 5$ kernel, ReLU activation.
- **MaxPooling Layer 1**: $2 \times 2$ pool size.
- **Conv2D Layer 2**: 64 filters, $3 \times 3$ kernel.
- **MaxPooling Layer 2**: $2 \times 2$ pool size.
- **Dense Layer**: 128 neurons with ReLU activation.
- **Output Layer**: 5 neurons with Softmax activation.

#### VGG16 Based Models
All VGG16 models utilize the pre-trained VGG16 convolutional base (ImageNet weights) with a custom classifier head:
- **VGG16 Base**: Convolutional layers for feature extraction.
- **Custom Head**: Flatten -> Dense (64, ReLU) -> Dropout (0.5) -> Dense (5, Softmax).
- **Transfer Learning Strategy**: All layers in the VGG16 base are frozen (trainable = False).
- **Fine-Tuning Strategy**: All layers except the last four layers of the VGG16 base are frozen, allowing deeper features to adapt to the dataset.

### 4. Data Augmentation Parameters
The augmented and VGG16 models use ImageDataGenerator for real-time transformations:
- **Rotation Range**: 40 degrees.
- **Width/Height Shift**: 20% of image dimensions.
- **Shear/Zoom**: 20% range.
- **Brightness Range**: 0.6 to 1.2.
- **Horizontal Flip**: Enabled.

## 💻 How to Load the Saved Models

You can load the pre-trained models using the following Python snippet:

```python
import joblib

# Example: Load the VGG16 Fine-Tuning Model
model = joblib.load('SavedModel/FineTuningVGG16AndUserDefineClassifier.joblib')

# Ensure your input data is normalized (1./255) and resized to (192, 192, 3)
```
