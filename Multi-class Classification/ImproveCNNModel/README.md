# ImproveCNNModel: Flower Classification & Data Augmentation

This project compares the classification performance of Convolutional Neural Networks (CNN) using the Flower Photos dataset. The primary focus is to evaluate how Data Augmentation techniques and Transfer Learning (using VGG16 and MobileNetV2) improve the model's accuracy and ability to generalize.

## 📁 Project Structure

The project is organized into the following directory structure:

```
ImproveCNNModel/
├── SavedModel/
│   ├── CNNModelWithImageDataGenerator.joblib
│   ├── CNNModelWithImageDataGeneratorAndDataArgmentation.joblib
│   ├── TransferLearningUsingVGG16AndUserDefineClassifier.joblib
│   ├── FineTuningVGG16AndUserDefineClassifier.joblib
│   └── TransferLearningUsingTensorFlowMobileNetAndUserDefineClassifier.h5
├── TrainingCode/
│   ├── CNNModelWithImageDataGenerator.py
│   ├── CNNModelWithImageDataGeneratorAndDataArgmentation.py
│   ├── TransferLearningUsingVGG16AndUserDefineClassifier.py
│   ├── FineTuningVGG16AndUserDefineClassifier.py
│   └── TransferLearningUsingTensorFlowMobileNetAndUserDefineClassifier.py
└── README.md
```

- **SavedModel/**: Contains serialized, pre-trained models.
- **TrainingCode/**: Contains the Python scripts for data preprocessing, CNN architecture definition, and the training pipeline.

## 📊 Model Performance Comparison

The following table shows the final validation accuracy for all models. The results demonstrate that MobileNetV2 and Fine-Tuned VGG16 provide the highest classification accuracy for this dataset.

| Model | Accuracy | Preprocessing / Strategy | Filename |
|-------|----------|---------------------------|----------|
| Standard CNN | 53.16% | Basic pixel normalization | CNNModelWithImageDataGenerator.joblib |
| Augmented CNN | 65.57% | Normalization + Data Augmentation | CNNModelWithImageDataGeneratorAndDataArgmentation.joblib |
| VGG16 Transfer Learning | 82.2% | Frozen VGG16 Base + Custom Classifier | TransferLearningUsingVGG16AndUserDefineClassifier.joblib |
| MobileNetV2 Transfer Learning | 91.14% | MobileNetV2 Fine-Tuning | TransferLearningUsingTensorFlowMobileNetAndUserDefineClassifier.h5 |
| VGG16 Fine-Tuning | 91.6% | Partial Unfreeze VGG16 + Custom Classifier | FineTuningVGG16AndUserDefineClassifier.joblib |

## 🛠️ Technical Details

### 1. Training Environment
- **Language**: Python 3.x
- **Core Libraries**: TensorFlow 2.x, Keras, TensorFlow Hub, Scikit-Learn, Joblib, Matplotlib

### 2. Dataset & Classes
The models utilize the Google flower_photos dataset:
- **Target Classes**: Daisy, Dandelion, Roses, Sunflowers, and Tulips.
- **Input Size**: RGB images resized to $192 \times 192$ pixels.

### 3. Model Architectures
#### VGG16 Based Models
- **VGG16 Base**: Uses ImageNet weights with a custom classifier head.
- **Classifier**: Flatten -> Dense (64, ReLU) -> Dropout (0.5) -> Dense (5, Softmax).

#### MobileNetV2 Model
- **Base Model**: MobileNetV2 (ImageNet weights) with pooling='avg' for automatic global average pooling.
- **Fine-Tuning**: The entire base model is set to trainable = True to allow full weight optimization during training.
- **Classifier**: Dense (64, ReLU) -> Dropout (0.5) -> Dense (5, Softmax).
- **Optimizer**: Adam with a learning rate of $1 \times 10^{-4}$.

### 4. Data Augmentation Parameters
All advanced models utilize ImageDataGenerator for real-time transformations:
- **Rotation Range**: 40 degrees.
- **Width/Height Shift**: 20% of image dimensions.
- **Shear/Zoom**: 20% range.
- **Brightness Range**: 0.6 to 1.2.
- **Horizontal Flip**: Enabled.

## 💻 How to Load the Saved Models

### Loading .joblib models (CNN & VGG16)
```python
import joblib
model = joblib.load('SavedModel/FineTuningVGG16AndUserDefineClassifier.joblib')
```

### Loading .h5 models (MobileNetV2)
```python
from tensorflow import keras
model = keras.models.load_model('SavedModel/TransferLearningUsingTensorFlowMobileNetAndUserDefineClassifier.h5')
```
