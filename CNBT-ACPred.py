import pandas as pd
import numpy as np
import csv
import os
import sys
from Bio import SeqIO
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Model

train_sequences = []
train_labels = []
for record in SeqIO.parse("Train_datasets_final.fasta", "fasta"):
    header = record.description
    sequence = str(record.seq)
    label = int(header.split("|")[1])
    train_sequences.append(sequence)
    train_labels.append(label)
train_sequences = np.array(train_sequences)
train_labels = np.array(train_labels)
amino_acid_map = {
    'X': 0, 'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
}

def encode_sequence(sequence, amino_acid_map):
    return [amino_acid_map[aa] for aa in sequence]
encoded_train_sequences = [encode_sequence(seq, amino_acid_map) for seq in train_sequences]
encoded_train_sequences = np.array(encoded_train_sequences)

test_sequences = []
for record in SeqIO.parse("final-uniprot_41-50-output-x.fasta", "fasta"):
    header = record.description
    sequence = str(record.seq)
    test_sequences.append(sequence)
test_sequences = np.array(test_sequences)

encoded_test_sequences = [encode_sequence(seq, amino_acid_map) for seq in test_sequences]
encoded_test_sequences = np.array(encoded_test_sequences)

cnn_model = tf.keras.Sequential([
    tf.keras.layers.Reshape((50, 1), input_shape=(50,)),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

cnn_model.fit(encoded_train_sequences, train_labels, epochs=25, batch_size=32, verbose=0)

cnn_y_pred_test = cnn_model.predict(encoded_test_sequences)
cnn_y_pred_flattened = cnn_y_pred_test.flatten()

num_amino_acids = 21
max_sequence_length = encoded_test_sequences.shape[1]
embedding_dim = 128
bi_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=num_amino_acids, output_dim=embedding_dim, input_length=max_sequence_length),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=128, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=64, return_sequences=False)),
    tf.keras.layers.Dropout(0),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])
bi_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

bi_model.fit(encoded_train_sequences, train_labels, epochs=25, batch_size=32, verbose=0)

bi_y_pred_test = bi_model.predict(encoded_test_sequences)

bi_y_pred_flattened = bi_y_pred_test.flatten()

max_length = encoded_train_sequences.shape[1]
vocab_size = len(amino_acid_map) + 1
def transformer_model(max_length, vocab_size, num_heads=6, embedding_dim=256, dense_units=128, dropout_rate=0):
    inputs = Input(shape=(max_length,))
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)
    x = TransformerEncoder(num_heads, embedding_dim, dense_units, dropout_rate, x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    trans_model = Model(inputs=inputs, outputs=outputs)
    return trans_model
def TransformerEncoder(num_heads, embedding_dim, dense_units, dropout_rate=0.2, inputs=None):
    x = PositionalEncoding()(inputs)
    x = Dropout(dropout_rate)(x)
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)
    x = attention(x, x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dense(dense_units, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    return x
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self):
        super(PositionalEncoding, self).__init__()
    def call(self, inputs):
        seq_length = inputs.shape.as_list()[-2]
        d_model = inputs.shape.as_list()[-1]
        pos = np.arange(seq_length)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rads = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        pos_encoding = angle_rads[np.newaxis, ...]
        return inputs + tf.cast(pos_encoding, dtype=tf.float32)

trans_model = transformer_model(max_length, vocab_size, num_heads=6, embedding_dim=embedding_dim)
trans_model.compile(optimizer='adam',
           loss='binary_crossentropy',
           metrics=['accuracy'])

trans_model.fit(encoded_train_sequences, train_labels, epochs=25, batch_size=32, verbose=0)

trans_y_pred_test = trans_model.predict(encoded_test_sequences)

trans_y_pred_flattened = trans_y_pred_test.flatten()

avg_predictions = (cnn_y_pred_flattened*1 + bi_y_pred_flattened*1 + trans_y_pred_flattened*1) / 3

predictions = {
    "Peptide Sequence": test_sequences,
    "CNN Prediction Probability": cnn_y_pred_flattened,
    "CNN_Bi-LSTM Prediction Probability": bi_y_pred_flattened,
    "Transformer Prediction Probability": trans_y_pred_flattened,
    "Avg Predictions": avg_predictions
}

results_df = pd.DataFrame(predictions)

results_df.to_csv("predictions_results_uniprot_21-30.csv", index=False)
