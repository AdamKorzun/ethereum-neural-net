import tensorflow as tf
import pandas as pd
import numpy as np
import os
import re
import string

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
np.set_printoptions(precision=3, suppress=True)
ethdata = pd.read_csv("data/eth_prices/eth_usd_Kraken.csv",skiprows = 1, index_col=False, names=["Time", "UnixTimestamp", "Symbol", "Open","High", "Close", "VolumeETH","VolumeUSD"])
#ethdata.pop('UnixTimestamp')
ethdata.pop('Symbol')
ethdata.pop('VolumeETH')
ethdata.pop('VolumeUSD')

class NeuralNetwork:
    raw_train_ds = None
    raw_test_ds = None
    encoder = None
    model = None
    batch_size=None
    epocs = None
    def load_data(self, article_data_path, batch_size, training_labels,testing_labels,seed, validation_split = 0.2):
        self.batch_size = batch_size
        self.raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
            article_data_path,
            batch_size=batch_size,
            labels=training_labels,
            validation_split=validation_split,
            subset='training',
            seed=seed)
        self.raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
            article_data_path,
            labels=testing_labels,
            batch_size=batch_size)
    def standardize_data(self, vocab_size):
        def custom_standardization(input_data):
            lowercase = tf.strings.lower(input_data)
            lowercase = tf.strings.unicode_decode(lowercase, 'UTF-8',errors='replace', replacement_char=32)
            lowercase = tf.strings.unicode_encode(lowercase, 'UTF-8',errors='replace', replacement_char=32)
            converted_string = tf.strings.regex_replace(lowercase, r'\n', ' ')
            return tf.strings.regex_replace(converted_string,
                                          '[%s]' % re.escape(string.punctuation),
                                          '')
        self.encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
            standardize=custom_standardization,
            max_tokens=vocab_size)
    def train_model(self, checkpoint_filepath, saved_model_filepath, epochs):
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        self.encoder.adapt(self.raw_train_ds.map(lambda text, label: text))
        self.model = tf.keras.Sequential([
            self.encoder,
            tf.keras.layers.Embedding(
                input_dim=len(self.encoder.get_vocabulary()),
                output_dim=64,
                mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        self.model.compile(loss=tf.keras.losses.MeanSquaredLogarithmicError(),
                      optimizer=tf.keras.optimizers.Adam(1e-4),
                      metrics=['MeanSquaredLogarithmicError'])
        self.model.fit(self.raw_train_ds, epochs=self.epochs,
                            validation_data=self.raw_test_ds,
                            callbacks=[model_checkpoint_callback],
                            steps_per_epoch = len(list(self.raw_train_ds)) / self.batch_size,
                            validation_steps=30)
        self.model.save(saved_model_filepath)


article_data_path = 'data/articles/'
checkpoint_filepath = 'model/checkpoint/'
saved_model_filepath = 'model/saved_model/'

if (not os.path.isdir(checkpoint_filepath)):
    os.mkdir(checkpoint_filepath)
if (not os.path.isdir(saved_model_filepath)):
    os.mkdir(saved_model_filepath)

def get_labels(file_path):
    timestamp_list = []
    price_change_list = []
    for file in os.listdir(file_path):
        timestamp = file.split('_')[1]
        timestamp = timestamp.split('.')[0]+'.'+timestamp.split('.')[1]
        timestamp_list.append(float(timestamp))
    for timestamp in timestamp_list:
        eth_row = ethdata.loc[(ethdata['Time'] <= timestamp)]
        #eth_row = eth_row.head(1)
        eth_row = eth_row.iloc[0]
        price_change = eth_row['Open']
        eth_row = ethdata.loc[(ethdata['Time'] >= timestamp)]
        try:
            eth_row = eth_row.iloc[23]
            price_change -= eth_row['Open']
        except Exception as e:
            price_change = 0
        price_change*=-1
        price_change_list.append(price_change)
    return price_change_list

vocab_size=1000
batch_size = 1
model = NeuralNetwork()
model.load_data(article_data_path, batch_size, get_labels(article_data_path + '/training'),get_labels(article_data_path + '/training'),42)
model.standardize_data(vocab_size)
model.train_model(checkpoint_filepath = 'model/checkpoint/', saved_model_filepath, epochs = 1)
