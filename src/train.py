import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from avro.datafile import DataFileReader
from avro.io import DatumReader


DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'takehome.avro')

# read data from the avro file, and generate a pandas
# dataframe out of it
def read_data():
    # read avro file into an array of dicts
    reader = DataFileReader(open(DATA_FILE_PATH, "rb"), DatumReader())

    try:
        data = []
        for row in reader:
            data.append(row)

        # pandas can only read json or csv
        # convert data to json object
        json_data = json.dumps(data)

        # read the json into a pandas dataframe
        dataset =  pd.read_json(json_data)

        # split into train and test data 
        train_dataset = dataset.sample(frac=0.8, random_state=0)
        test_dataset = dataset.drop(train_dataset.index)

        # separate features and labels
        train_features = train_dataset.copy().drop('rating', 1).to_numpy()
        test_features = test_dataset.copy().drop('rating', 1).to_numpy()

        train_labels = left_shift_labels(train_dataset.copy().pop('rating'))
        test_labels = left_shift_labels(train_dataset.copy().pop('rating'))

        return train_features, train_labels, test_features, test_labels
    finally:
        reader.close()

# takes in a pandas series containing labels
# from 1 to 10, and returns a numpy array containing
# of the same labels, just left shifted by one, in 
# the range 0 to 9
def left_shift_labels(labels):
    labels = labels.to_numpy()
    labels -= 1
    return labels

def train():
    train_features, train_labels, test_features, test_labels = read_data()
    print(train_features)
    print(train_features.shape)
    print(train_labels)
    print(train_labels.shape)

    num_features = train_features.shape[1]
    print(num_features)

    # train
    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=(num_features,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.000001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    model.fit(train_features, train_labels, epochs=100)
    print(model.predict(train_features[:10]))

if __name__ == '__main__':
    train()
