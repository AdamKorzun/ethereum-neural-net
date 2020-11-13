import tensorflow as tf
import pandas as pd
import numpy as np
import os
import re
import string
import random


from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


print(tf.__version__)

class EtherNN(tf.keras.Model):
    training_data = None
    testing_data = None
    model_checkpoint_callback = None
    def __init__(self, vocab_size, checkpoint_filepath = None):
        super(EtherNN, self).__init__()
        if (checkpoint_filepath):
            self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='val_loss',
                mode='min')
        def custom_standardization(input_data):
            lowercase = tf.strings.lower(input_data)
            lowercase = tf.strings.unicode_decode(lowercase, 'UTF-8',errors='replace', replacement_char=32)
            lowercase = tf.strings.unicode_encode(lowercase, 'UTF-8',errors='replace', replacement_char=32)
            converted_string = tf.strings.regex_replace(lowercase, r'\n', '')
            return tf.strings.regex_replace(converted_string,
                                          '[%s]' % re.escape(string.punctuation),
                                          '')
        self.encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
            standardize=custom_standardization,
            max_tokens=vocab_size)
        self.EmbleddingLayer = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=64,
            mask_zero=True)
        self.BidirectionalLayer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))
        self.DenseL1 = tf.keras.layers.Dense(64, activation='relu')
        self.DenseL2 = tf.keras.layers.Dense(1)

    def call(self, input_data, training=False):
        x = self.encoder(input_data)
        x = self.EmbleddingLayer(x)
        x = self.BidirectionalLayer(x)
        x = self.DenseL1(x)
        return self.DenseL2(x)

    def load_data(self, article_data_path, ethereum_prices_path, batch_size, validation_split = 0.2):
        ethdata = pd.read_csv(ethereum_prices_path,
                            skiprows = 1,
                            index_col=False,
                            names=["Time", "UnixTimestamp", "Symbol", "Open","High", "Close", "VolumeETH","VolumeUSD"])
        #ethdata.pop('UnixTimestamp')
        ethdata.pop('Symbol')
        ethdata.pop('VolumeETH')
        ethdata.pop('VolumeUSD')
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
        seed = random.randint(0,1000)
        self.training_data = tf.keras.preprocessing.text_dataset_from_directory(
            article_data_path,
            batch_size=batch_size,
            labels=get_labels(article_data_path + '/training'),
            validation_split=validation_split,
            subset='training',
            seed=seed)
        self.testing_data = tf.keras.preprocessing.text_dataset_from_directory(
            article_data_path,
            labels=get_labels(article_data_path + '/training'),
            batch_size=batch_size)

        self.EmbleddingLayer = tf.keras.layers.Embedding(
            input_dim=len(list(self.training_data)),
            output_dim=64,
            mask_zero=True)
        #self.encoder.adapt(self.training_data.map(lambda text, label: text))
    def get_config(self):
        None

article_data_path = 'data/articles/'
checkpoint_filepath = 'model/checkpoint/'
saved_model_filepath = 'model/saved_model/'
ethereum_prices_path = 'data/eth_prices/eth_usd_Kraken.csv'
if (not os.path.isdir(checkpoint_filepath)):
    os.mkdir(checkpoint_filepath)
if (not os.path.isdir(saved_model_filepath)):
    os.mkdir(saved_model_filepath)


vocab_size=1000
batch_size = 1


# model =tf.keras.models.load_model("model\\saved_model\\",custom_objects={'TextVectorization' : aencoder,'custom_standardization':custom_standardization})
# model = EtherNN(vocab_size, checkpoint_filepath)
# model.load_weights(checkpoint_filepath)
# with open('data/articles/training/60-of-ethereum-nodes-not-ready-for-istanbul-blockchain-upgrade_1575406819.0.txt','r') as file:
#     print(model.predict([file.read()]))
model = EtherNN(vocab_size, checkpoint_filepath)

model.load_data(article_data_path, ethereum_prices_path, batch_size)
model.compile(loss=tf.keras.losses.MeanSquaredLogarithmicError(),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['MeanSquaredLogarithmicError'])

model.fit(model.training_data,
        epochs=100,
        validation_data=model.testing_data,
        callbacks=[model.model_checkpoint_callback],
        steps_per_epoch=len(list(model.training_data)),
        validation_steps = 30)
# model.save(saved_model_filepath)
