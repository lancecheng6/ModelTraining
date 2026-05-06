# Import all dependencies for this chapter
import collections
import operator
import random
from typing import List, Dict

import numpy as np
import pandas as pd
from hanziconv import HanziConv
from segtok.tokenizer import word_tokenizer
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# 1. Define class
class Processor(object):

    def build_token_dict(self, corpus: List[List[str]]):
        """
        Build token dictionary. This method will iterate through the tokenized corpus 
        to build a token frequency dictionary and a mapping dictionary from tokens to indices.

        Args:
            corpus: All tokenized corpus
        """
        token2idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }

        token2count = {}
        for sentence in corpus:
            for token in sentence:
                count = token2count.get(token, 0)
                token2count[token] = count + 1
        # Sort by token frequency in descending order
        sorted_token2count = sorted(token2count.items(),
                                    key=operator.itemgetter(1),
                                    reverse=True)
        token2count = collections.OrderedDict(sorted_token2count)

        for token in token2count.keys():
            if token not in token2idx:
                token2idx[token] = len(token2idx)
        return token2idx, token2count

    @staticmethod
    def numerize_sequences(sequence: List[str],
                           token2index: Dict[str, int]) -> List[int]:
        """
        Convert an array of tokenized marks (tokens) into a corresponding array of indices.
        e.g., ['I', 'want', 'sleep'] -> [10, 313, 233]

        Args:
            sequence: Tokenized mark array
            token2index: Index dictionary
        Returns: Index array corresponding to the input data
        """
        token_result = []
        for token in sequence:
            token_index = token2index.get(token)
            if token_index is None:
                token_index = token2index['<UNK>']
            token_result.append(token_index)
        return token_result

data_path = '/kaggle/input/datasets/lancecheng666888/cmn-eng2/cmn.txt'
df = pd.read_csv(data_path, header=None, sep='\t')

# Add table headers
df.columns = ['en', 'cn', 'cc']
# Take the first 5000 records
df = df[:5000]

# Convert Simplified Chinese to Traditional Chinese
df['cn'] = df['cn'].apply(lambda x: HanziConv.toTraditional(x))


# Use segtok for tokenization; convert all text to lowercase before tokenizing
df['en_cutted'] = df['en'].apply(lambda x: word_tokenizer(x.lower()))
# Character-based tokenization, while adding start and end markers
df['cn_cutted'] = df['cn'].apply(lambda x: ['<BOS>'] + list(x) + ['<EOS>'])
#df.head()

p = Processor()

p.input2idx, p.input2count = p.build_token_dict(df.en_cutted.to_list())
p.output2idx, p.output2count = p.build_token_dict(df.cn_cutted.to_list())

p.idx2output = dict([(v, k) for k, v in p.output2idx.items()])

ENCODER_DIM = len(p.input2idx)
DECODER_DIM = len(p.output2idx)

# Read sequence lengths
EN_SEQ_LEN = max([len(seq) for seq in df.en_cutted.to_list()])
CN_SEQ_LEN = max([len(seq) for seq in df.cn_cutted.to_list()])

# Number of hidden layer dimensions
HIDDEN_LAYER_DIM = 512

tokenized_en = []
tokenized_cn = []

for input_seq in df.en_cutted.to_list():
    tokenized_en.append(p.numerize_sequences(input_seq, p.input2idx))

for output_seq in df.cn_cutted.to_list():
    tokenized_cn.append(p.numerize_sequences(output_seq, p.output2idx))

padded_en = pad_sequences(tokenized_en, EN_SEQ_LEN, padding='post', truncating='post')
padded_cn = pad_sequences(tokenized_cn, CN_SEQ_LEN, padding='post', truncating='post')

encoder_input_data = padded_en
# Use sequences from time step 0 to the second-to-last time step as decoder input
decoder_input_data = padded_cn[:, :-1]
# Use sequences from time step 1 to the last time step as decoder target
# Since the output layer calculates loss via cross-entropy, the decoder output needs to be one-hot encoded
decoder_output_data = to_categorical(padded_cn[:, 1:], DECODER_DIM)

# 2. Define the training model
L = keras.layers

# Encoder input
encoder_inputs = L.Input(shape=(None,),
                         name='encoder_inputs')

# Encoder Embedding layer
encoder_embedding_layer = L.Embedding(input_dim=ENCODER_DIM,
                                      output_dim=64,
                                      name='encoder_embedding')

# Encoder LSTM layer
encoder_lstm_layer = L.LSTM(HIDDEN_LAYER_DIM,
                            return_state=True,  # Return the hidden state of the encoder
                            name='encoder_lstm')

encoder_embeddings = encoder_embedding_layer(encoder_inputs)
# Get the hidden state of the encoder LSTM layer
encoder_outputs, state_h, state_c = encoder_lstm_layer(encoder_embeddings)

encoder_states = [state_h, state_c]

# Decoder input
decoder_inputs = L.Input(shape=(None,),
                         name='decoder_inputs')
# Decoder Embedding layer
decoder_embedding_layer = L.Embedding(input_dim=DECODER_DIM,
                                      output_dim=64,
                                      name='decoder_embedding')
# Decoder LSTM layer
decoder_lstm_layer = L.LSTM(HIDDEN_LAYER_DIM,
                            return_sequences=True,  # Return sequences
                            return_state=True,  # Return the hidden state
                            name='decoder_lstm')

# Decoder fully connected output layer
decoder_dense_layer = L.Dense(DECODER_DIM,
                              activation='softmax',
                              name='decoder_dense')

decoder_embeddings = decoder_embedding_layer(decoder_inputs)
# Use encoder hidden states as the initial state for the decoder
decoder_lstm_output, state_h, state_c = decoder_lstm_layer(decoder_embeddings,
                                                           initial_state=encoder_states)
decoder_outputs = decoder_dense_layer(decoder_lstm_output)

# Construct model: inputs are encoder and decoder inputs, output is decoder output
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.summary()

# 3. Train model
# Train for 100 epochs
model.fit([encoder_input_data, decoder_input_data],
          decoder_output_data,
          epochs=100,
          batch_size=64,
          callbacks=[])

# 4. Define prediction model
# Encoder model input is "encoder_inputs", output is "encoder hidden states"
encoder_model = keras.Model(encoder_inputs, encoder_states)
encoder_model.summary()

# Decoder model accepts "decoder embedding results" and "previous hidden states" as inputs
decoder_state_input_h = L.Input(shape=(HIDDEN_LAYER_DIM,))
decoder_state_input_c = L.Input(shape=(HIDDEN_LAYER_DIM,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

decoder_lstm_outputs, h, c = decoder_lstm_layer(decoder_embeddings,
                                                initial_state=decoder_states_inputs)

# Decoder outputs the "target sequence" and "current hidden states"
decoder_states = [h, c]
decoder_outputs = decoder_dense_layer(decoder_lstm_outputs)

decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs,
                            [decoder_outputs] + decoder_states)
decoder_model.summary()


def translate_sentence(sentence: List[str]):
    """
    Translate sentence
    Args:
        sentence: Original sentence

    Returns:
        Translation result
    """
    # Convert input sentence to idx sequence and pad the sequence
    vec_sen = p.numerize_sequences(sentence, p.input2idx)
    vec_sen = pad_sequences([vec_sen], EN_SEQ_LEN, padding='post', truncating='post')
    # Get Thought Vector
    h1, c1 = encoder_model.predict(vec_sen)

    # Start predicting with the start marker <BOS> as the input token
    target_seq = np.array([[p.output2idx['<BOS>']]])

    outputs: List[int] = []

    while True:
        # Predict the next token and update hidden states
        output_tokens, h1, c1 = decoder_model.predict([target_seq, h1, c1])
        # Use argmax to get the ID of the next token
        sampled_token_index: int = np.argmax(output_tokens[0, -1, :])

        # Stop predicting when the end marker is reached or the sequence is too long
        if sampled_token_index == p.output2idx['<EOS>'] or len(outputs) > 30:
            break

        outputs.append(sampled_token_index)
        # Use the predicted token as the next input
        target_seq = np.array([[sampled_token_index]])

    return ''.join([p.idx2output[output] for output in outputs])

# 5. Randomly pick 10 English sentences and translate them into Chinese
for i in range(10):
    sentence = random.choice(df.en_cutted.to_list())
    res = translate_sentence(sentence)
    print(f"{' '.join(sentence):30}-> {res}")
    
# 6. Save Model
# i. Save the entire training model
model.save('/kaggle/working/outputs/s2s_train_model.h5')

# ii. Save prediction models separately
encoder_model.save('/kaggle/working/outputs/encoder_model.h5')
decoder_model.save('/kaggle/working/outputs/decoder_model.h5')

# iii. Save Processor (Dictionary)
# The model only recognizes numbers (indices); without the dictionary, 
# you cannot map those numbers back into characters/text.
import pickle
with open('/kaggle/working/outputs/processor.pkl', 'wb') as f:
    pickle.dump(p, f)

# 7. Load Model
#from tensorflow.keras.models import load_model
#
## Load models specific to inference/prediction
#encoder_model = load_model('/kaggle/working/outputs/encoder_model.h5')
#decoder_model = load_model('/kaggle/working/outputs/decoder_model.h5')
#
## Load dictionary
#with open('/kaggle/working/outputs/processor.pkl', 'rb') as f:
#    p = pickle.load(f)