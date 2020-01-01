import numpy as np
import csv as csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation

def preprocess_data(x, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(x)
    x = scaler.transform(x)
    return x, scaler

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def prepare_data(shuffle=False):
    x_train = list()
    with open("./data/train.csv") as f:
        x_reader = csv.reader(f, delimiter=";")
        for row in x_reader:
            x_train.append(row)
    x_train = np.array(x_train)

    x_test = list()
    with open("./data/test.csv") as f:
        x_reader = csv.reader(f, delimiter=";")
        for row in x_reader:
            x_test.append(row)
    x_test = np.array(x_test)

    y_index = np.where(x_train[0,:] == "Target")[0][0]
    y_train = x_train[1:,y_index].astype(np.float32)-1
    x_train = np.delete(x_train, y_index, axis=1)[1:,:].astype(np.float32)

    x_test = x_test[1:,:].astype(np.float32)

    if (shuffle):
        rng_state = np.random.get_state()
        np.random.shuffle(x_train)
        np.random.set_state(rng_state)
        np.random.shuffle(y_train)
        rng_state = np.random.get_state()
        np.random.shuffle(x_test)

    x_train, scaler = preprocess_data(x_train)
    y_train, encoder = preprocess_labels(y_train)

    x_test, _ = preprocess_data(x_test, scaler)

    return x_train, y_train, x_test

def train():
    x_train, y_train, x_test = prepare_data(shuffle=True)

    dims = x_train.shape[1]
    nb_classes = y_train.shape[1]

    model = Sequential()
    model.add(Dense(128, input_shape=(dims,), activation = "relu"))
    model.add(Dense(64, activation = "relu"))
    model.add(Dense(32, activation = "relu"))
    model.add(Dense(nb_classes, activation = "softmax"))

    model.compile(optimizer='adam', loss='categorical_hinge', metrics=['accuracy'])

    n_epochs = 80

    network_history = model.fit(x_train, y_train, batch_size=256, epochs=n_epochs, verbose=1, validation_split=0.2)