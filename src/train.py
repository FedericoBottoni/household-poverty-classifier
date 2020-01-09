import math
import numpy as np
import csv as csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
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

def prepare_data(split_size=0.25, shuffle=False):
    x = list()
    with open("./data/train.csv") as f:
        x_reader = csv.reader(f, delimiter=";")
        for row in x_reader:
            x.append(row)
    x = np.array(x)

    y_index = np.where(x[0,:] == "Target")[0][0]
    y = x[1:,y_index].astype(np.float32)-1 #-1 is used to scale the target column to 0-base
    x = np.delete(x, y_index, axis=1)[1:,:].astype(np.float32)

    if (shuffle):
        rng_state = np.random.get_state()
        np.random.shuffle(x)
        np.random.set_state(rng_state)
        np.random.shuffle(y)

    x, _ = preprocess_data(x)

    r_size = x.shape[0]
    x_train = x[:math.floor(r_size*(1-split_size)),:]
    x_test = x[math.floor(r_size*(1-split_size)):,:]
    y_train = y[:math.floor(r_size*(1-split_size))]
    y_test = y[math.floor(r_size*(1-split_size)):]

    y_train, _ = preprocess_labels(y_train)

    return x_train, y_train, x_test, y_test

def train():
    x_train, y_train, x_test, y_test = prepare_data(split_size=0.1, shuffle=True)

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

    model.summary()

    labels=model.predict_classes(x_test, batch_size=256, verbose=1)
    print("Accuracy: ", accuracy_score(labels, y_test))
    print(classification_report(labels, y_test))