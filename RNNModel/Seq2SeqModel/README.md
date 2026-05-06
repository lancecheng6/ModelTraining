Seq2SeqModel/
├── TrainingCode/
│   └── EnglishToChineseTranslation(Seq2SeqModel).py
├── TrainingData/
│   └── cmn.txt
└── README.md


- **TrainingCode/**: Contains the Python script for data preprocessing, Seq2Seq model definition, training, and inference.
- **TrainingData/**: Contains the source dataset (`cmn.txt`) consisting of English-Chinese sentence pairs.
- **README.md**: Documentation for the project.

## 📊 Model Highlights

The model is trained on a subset of English-Chinese pairs and evaluated by randomly selecting sentences for translation. Key features include:

- **Architecture**: Neural Machine Translation using an Encoder-Decoder LSTM framework.
- **Translation Task**: Mapping English word sequences to Chinese character sequences.
- **Preprocessing**: Handles text normalization, tokenization, and conversion from Simplified to Traditional Chinese.

## 🛠️ Technical Details

### 1. Training Environment
- **Platform**: Python 3.x
- **Key Libraries**: TensorFlow/Keras, NumPy, Pandas, HanziConv (Chinese conversion), Segtok (English tokenization)

### 2. Dataset & Preprocessing
The model utilizes the `cmn.txt` dataset, focusing on the first 5,000 records for demonstration:
- **English (Source)**: Converted to lowercase and tokenized using `word_tokenizer`.
- **Chinese (Target)**: Converted to Traditional Chinese and tokenized at the character level, with `<BOS>` (Beginning of Sentence) and `<EOS>` (End of Sentence) markers added.
- **Sequence Handling**: Sequences are padded to a fixed length using "post" padding to ensure uniform input shapes.

### 3. Model Architecture
- **Encoder**: 
    - Input: English sequences.
    - Embedding Layer: Output dimension of 64.
    - LSTM Layer: 512 hidden units; returns internal states (hidden and cell states) as the "Thought Vector".
- **Decoder**:
    - Input: Chinese sequences (shifted by one time step for Teacher Forcing).
    - Embedding Layer: Output dimension of 64.
    - LSTM Layer: 512 hidden units; initialized with the encoder's hidden states.
    - Dense Output Layer: Softmax activation over the Chinese vocabulary.

### 4. Training Configuration
- **Loss Function**: Categorical Crossentropy.
- **Optimizer**: RMSprop.
- **Epochs**: 100.
- **Batch Size**: 64.

## 💻 How to Use

### Running Training and Inference
To train the model and see sample translations, execute the main script:

python TrainingCode/EnglishToChineseTranslation(Seq2SeqModel).py

