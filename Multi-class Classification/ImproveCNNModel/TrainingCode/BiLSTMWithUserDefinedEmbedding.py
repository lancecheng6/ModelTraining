import jieba
import pandas as pd
import matplotlib.pyplot as plt
import collections
import operator
import os
import json
import gensim
import numpy as np
import pathlib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List

%matplotlib inline  

# 1. define function
def read_data_as_pd(file_path: str) -> pd.DataFrame:
    """Read dataset as DataFrame format

    Args:
        file_path: Original file path
    Returns:
        Dataset DataFrame
    """
    json_data = json.load(open(file_path, 'r'))
    value_list = list(json_data.values())
    return pd.DataFrame(value_list)

def process_query(query: str) -> List[str]:
    """ Preprocess query, e.g., 'I want to see a movie' -> ['I', 'want', 'to', 'see', 'a', 'movie']

    Args:
        query: query text
    Returns:
        Array of processed tokens
    """
    stripped_query = query.strip()
    return list(jieba.cut(stripped_query))

def visualize_train_process(history: tf.keras.callbacks.History,
                            title: str):
    plt.figure()
    # Set subplot size
    plt.subplots(figsize=(10,4))
    for index, target in enumerate(['accuracy', 'loss']):
        plt.subplot(1, 2, index + 1)
        plt.plot(history.history[target], label=target)
        plt.plot(history.history[f'val_{target}'], label=f'val_{target}')
        plt.legend()
        plt.title(f'{title} {target}')
    plt.show()

# 2. define class
class Processor(object):

    def __init__(self):
        self.token2idx = {}                          # Token index dictionary
        self.token2count = collections.OrderedDict() # Token frequency table
        self.label2idx = {}                          # Label index dictionary
        self.idx2label = {}                          # Index to label dictionary

    def build_token_dict(self, corpus: List[List[str]]):
        """
        Build token dictionary. This method iterates through the tokenized corpus, 
        building a token frequency dictionary and a mapping dictionary between tokens and indices.
        Args:
            corpus: All tokenized corpus
        """
        token2idx = {
            '<PAD>': 0,
            '<UNK>': 1
        }

        token2count = {}
        for sentence in corpus:
            for token in sentence:
                count = token2count.get(token, 0)
                token2count[token] = count + 1
        # Sort by frequency in descending order
        sorted_token2count = sorted(token2count.items(),
                                    key=operator.itemgetter(1),
                                    reverse=True)
        self.token2count = collections.OrderedDict(sorted_token2count)

        for token in self.token2count.keys():
            token2idx[token] = len(token2idx)
        self.token2idx = token2idx

    def build_label_dict(self, labels: List[str]):
        """
        Build label index mapping dictionary
        Args:
            labels: Labels corresponding to all corpus data
        """
        label2idx = {}
        for label in labels:
            if label not in label2idx:
                label2idx[label] = len(label2idx)
        self.label2idx = label2idx
        self.idx2label = dict([(index, label) for label, index in label2idx.items()])

    def convert_text_to_index(self, sentence: List[str]):
        """
        Convert token array to corresponding index array
        E.g., ['I', 'want', 'sleep'] -> [10, 313, 233]
        Args:
            sentence: Tokenized array
        Returns: Index array corresponding to input data
        """
        token_result = []
        for token in sentence:
            token_result.append(self.token2idx.get(token, self.token2idx['<UNK>']))
        return token_result
    
    def build_from_w2v(self, w2v_path: str):
        """
        Build vocabulary and word vector table using pre-trained word embeddings
        Args:
            w2v_path: Path to pre-trained word embedding file
        """
        w2v = gensim.models.KeyedVectors.load_word2vec_format(w2v_path)

        token2idx = {
            '<PAD>': 0, # Since we pad sequences with 0, the padding token index must be 0
            '<UNK>': 1  # The unknown token index can be any value; 1 is set for convenience
        }

        # Iterate through pre-trained embedding vocabulary and add to our token index dictionary
        for token in w2v.index_to_key:
            token2idx[token] = len(token2idx)

        # Initialize a zero tensor of shape [total tokens, pre-trained vector dimension]
        vector_matrix = np.zeros((len(token2idx), w2v.vector_size))
        # Randomly initialize tensor for <UNK> token
        vector_matrix[1] = np.random.rand(300)
        # Use pre-trained vectors starting from index 2
        vector_matrix[2:] = w2v.vectors
        self.w2v = w2v
        self.vector_matrix = vector_matrix
        self.token2idx = token2idx
        
    def save_processor(self, folder: str):
        """
        Save Processor information to target folder
        Args:
            folder: Target folder path
        """
        pathlib.Path(folder).mkdir(exist_ok=True, parents=True)
        token_index_path = os.path.join(folder, 'token_index.json')
        with open(token_index_path, 'w') as f:
            f.write(json.dumps(self.token2idx, ensure_ascii=False, indent=2))

        label_index_path = os.path.join(folder, 'label_index.json')
        with open(label_index_path, 'w') as f:
            f.write(json.dumps(self.label2idx, ensure_ascii=False, indent=2))

    def load_processor(self, folder: str):
        """
        Load saved Processor
        Args:
            folder: Target folder path
        """
        token_index_path = os.path.join(folder, 'token_index.json')
        with open(token_index_path, 'r') as f:
            self.token2idx = json.load(f)

        label_index_path = os.path.join(folder, 'label_index.json')
        with open(label_index_path, 'r') as f:
            self.label2idx = json.load(f)
            self.idx2label = dict([(v, k) for k, v in self.label2idx.items()])

plt.rcParams['figure.dpi'] = 180
plt.rcParams['axes.grid'] = False

train_df = read_data_as_pd('/kaggle/input/datasets/lancecheng666888/smp2018/train.json')
test_df = read_data_as_pd('/kaggle/input/datasets/lancecheng666888/smp2018/dev.json')

train_df['cutted'] = train_df['query'].apply(process_query)
test_df['cutted'] = test_df['query'].apply(process_query)

#train_df.head()
#train_df['cutted'].apply(lambda x: len(x)).hist()

processor = Processor()
processor.build_token_dict(list(train_df.cutted) + list(test_df.cutted))
processor.build_label_dict(list(train_df.label) + list(test_df.label))

# Convert tokenized queries to corresponding index arrays
train_x = [processor.convert_text_to_index(query) for query in list(train_df.cutted)]
test_x  = [processor.convert_text_to_index(query) for query in list(test_df.cutted)]

# Pad sequences to a uniform length
train_x = pad_sequences(train_x, maxlen=15)
test_x  = pad_sequences(test_x, maxlen=15)

# Convert labels to corresponding indices
train_y = np.array([processor.label2idx[label] for label in list(train_df.label)])
test_y  = np.array([processor.label2idx[label] for label in list(test_df.label)])

#print(train_x[:5])
#print(train_y[:5])

# 3. Build Model ( UserDefinedEmbedding + BiLSTM )
L = tf.keras.layers

model = tf.keras.Sequential([
    # Use Embedding layer, input dimension equals vocabulary size
    L.Embedding(input_dim=len(processor.token2idx),
                output_dim=100,
                input_shape=(15,)),
    # Bidirectional LSTM
    L.Bidirectional(L.LSTM(64)),
    # Fully connected layer
    L.Dense(64, activation=tf.nn.relu),
    # Last fully connected layer output dimension equals label count
    L.Dense(len(processor.label2idx), activation=tf.nn.softmax)
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

hist = model.fit(train_x,
                 train_y,
                 validation_split=0.15,
                 epochs=20)

test_loss, test_acc = model.evaluate(test_x, test_y, verbose=0)
print(f'test loss: {test_loss}, test accuracy: {test_acc}')

visualize_train_process(hist, 'BiLSTM')

# 4. Save Model
model.save('/kaggle/working/outputs/USDFEnbedding_model.h5')

# 5. Save Processor
processor.save_processor('/kaggle/working/outputs/processorUSDFEnbedding')

# Load Processor
#loaded_processor = Processor()
#loaded_processor.load_processor('/kaggle/working/outputs/processorUSDFEnbedding')
#
# Load Model
# Define a custom object mapping to tell Keras to use standard softmax when encountering 'softmax_v2'
#custom_objects = {'softmax_v2': tf.keras.activations.softmax}
# Pass custom_objects during load_model
#loaded_model = tf.keras.models.load_model(
#    '/kaggle/working/outputs/USDFEnbedding_model.h5', 
#    custom_objects=custom_objects
#)
#loaded_model.summary()