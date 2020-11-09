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

npethdata = np.array(ethdata)
article_data_path = 'data/articles/'
# article_data = []
#
# for file_path in os.listdir(article_data_path):
#     file = open(article_data_path + file_path,'r')
#     timestamp = file_path.split('_')[1]
#     timestamp = timestamp.split('.')[0]+'.'+timestamp.split('.')[1]
#     article_data.append([file.read(),timestamp])
#
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


batch_size = 1
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    article_data_path,
    batch_size=batch_size,
    labels=get_labels('data/articles/training'),
    validation_split=0.2,
    subset='training',
    seed=seed)

raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    article_data_path,
    labels=get_labels('data/articles/training'),
    batch_size=batch_size)

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    lowercase = tf.strings.unicode_decode(lowercase, 'UTF-8',errors='replace', replacement_char=32)
    lowercase = tf.strings.unicode_encode(lowercase, 'UTF-8',errors='replace', replacement_char=32)
    converted_string = tf.strings.regex_replace(lowercase, r'\n', ' ')
    return tf.strings.regex_replace(converted_string,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

BUFFER_SIZE = 10000

train_dataset = raw_train_ds.shuffle(BUFFER_SIZE).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test_dataset = raw_test_ds.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
VOCAB_SIZE=1000

encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    standardize=custom_standardization,
    max_tokens=VOCAB_SIZE)

encoder.adapt(raw_train_ds.map(lambda text, label: text))

model = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=64,
        mask_zero=True),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.MeanSquaredLogarithmicError(),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['MeanSquaredLogarithmicError'])
history = model.fit(raw_train_ds, epochs=100,
                    validation_data=raw_test_ds,
                    steps_per_epoch = len(list(raw_train_ds)) / batch_size,
                    validation_steps=30)
