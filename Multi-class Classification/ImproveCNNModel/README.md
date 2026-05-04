# Deep Learning Project: Image Classification & NLP Intent Recognition

This project explores two main domains of deep learning: Image Classification using Convolutional Neural Networks (CNNs) and Natural Language Processing (NLP) using Bidirectional LSTMs (BiLSTMs). It provides a comprehensive comparison between various architectures and embedding strategies.

## 📁 Project Structure

The repository is organized as follows:

```
ImproveCNNModel/
├── TrainingCode/
│   ├── CNNModelWithImageDataGenerator.py
│   ├── CNNModelWithImageDataGeneratorAndDataArgmentation.py
│   ├── TransferLearningUsingVGG16AndUserDefineClassifier.py
│   ├── FineTuningVGG16AndUserDefineClassifier.py
│   ├── TransferLearningUsingTensorFlowMobileNetAndUserDefineClassifier.py
│   ├── BiLSTMWithUserDefinedEmbedding.py
│   └── BiLSTMWithUserPreTrainedEmbedding.py
└── README.md
```

- **TrainingCode/**: Contains Python scripts for data preprocessing, model architecture definitions, and the training pipelines for both image and text classification.

## 📊 Performance Comparison

The following table summarizes the validation accuracy across different models. The results highlight the effectiveness of Transfer Learning in image tasks and Pre-trained Embeddings in NLP tasks.

| Category | Model | Accuracy | Strategy / Preprocessing |
|----------|-------|----------|--------------------------|
| **Image (CNN)** | Standard CNN | 53.16% | Basic pixel normalization |
| | Augmented CNN | 65.57% | Normalization + Data Augmentation |
| | VGG16 Transfer Learning | 82.2% | Frozen VGG16 Base + Custom Classifier |
| | MobileNetV2 Transfer Learning | 91.14% | MobileNetV2 Fine-Tuning |
| | VGG16 Fine-Tuning | 91.6% | Partial Unfreeze VGG16 + Custom Classifier |
| **NLP (BiLSTM)** | **BiLSTM (User-Defined)** | **75.07%** | **Learned Embedding layer (100d)** |
| | **BiLSTM (Pre-trained)** | **88.70%** | **Gensim Word2Vec (sgns.weibo.bigram)** |

## 🛠️ Technical Details

### 1. Development Environment
- **Language**: Python 3.x
- **Frameworks**: TensorFlow 2.x, Keras
- **Key Libraries**: Gensim (Word2Vec), Jieba (Chinese Segmentation), Scikit-Learn, Matplotlib

### 2. Natural Language Processing (BiLSTM)
The project implements intent recognition using Bidirectional LSTMs to capture bidirectional context in sentences:
- **Segmentation**: Uses `jieba` for Chinese word segmentation.
- **Embedding Strategies**:
    - **User-Defined**: An embedding layer initialized from scratch and trained alongside the model.
    - **Pre-trained**: Utilizes pre-trained Word2Vec vectors (trained on Weibo bigrams) to leverage existing semantic knowledge, resulting in a significant accuracy boost (~13.6%).
- **Architecture**: `Embedding` -> `Bidirectional(LSTM(64))` -> `Dense(64, ReLU)` -> `Dense(N, Softmax)`.

### 3. Image Classification (CNN)
The image models utilize the Google `flower_photos` dataset:
- **Transfer Learning**: Employs VGG16 and MobileNetV2 architectures pre-trained on ImageNet.
- **Data Augmentation**: Uses `ImageDataGenerator` for real-time transformations including rotation, zoom, shear, and flips to improve generalization.

## ⚠️ Important Note on Model Files
To keep the repository lightweight and comply with GitHub's file size limits, pre-trained model files (.h5 and .joblib) are **not included**. To reproduce the results, please run the corresponding scripts in the `TrainingCode/` directory.
