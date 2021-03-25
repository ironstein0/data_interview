import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from avro.datafile import DataFileReader
from avro.io import DatumReader
from sklearn import preprocessing


DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'takehome.avro')

# read data from the avro file, and generate a train and
# test features and labels out of it.
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

        # separate features and labels
        features = dataset.copy().drop('rating', 1)
        labels = dataset.copy().pop('rating')

        # normalize features
        features = normalize_features(features)

        # split into train and test data 
        train_features = features.sample(frac=0.8, random_state=0)
        test_features = features.drop(train_features.index)

        train_labels = labels[labels.index.isin(train_features.index)]
        test_labels = labels.drop(train_features.index)

        # convert features to numpy arrays
        train_features = train_features.to_numpy()
        test_features = test_features.to_numpy()

        # left shift labels to convert them from the range [1,10]
        # to the range [0, 9]
        train_labels = left_shift_labels(train_labels)
        test_labels = left_shift_labels(test_labels)

        return train_features, train_labels, test_features, test_labels
    finally:
        reader.close()

# normalizes all features to the range [0, 1]
def normalize_features(dataset):
    dataset = dataset.values
    min_max_scaler = preprocessing.MinMaxScaler()
    dataset_scaled = min_max_scaler.fit_transform(dataset)
    dataset = pd.DataFrame(dataset_scaled)
    return dataset

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
        layers.Dense(12, activation='relu'),
        layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.000001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    model.fit(train_features, train_labels, epochs=10)
    print(model.predict(test_features[:10]))

if __name__ == '__main__':
    train()
