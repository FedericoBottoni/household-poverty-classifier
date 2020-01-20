import math
import numpy as np
import csv as csv
from functools import partial
from collections import defaultdict
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout
from score_helper import leave1out_cv
from plot import plot_history

def shuffle_dataset(xs, ys):
    idx = np.random.permutation(xs.shape[0])
    xs_result, ys_result = xs[idx], ys[idx]
    return xs_result, ys_result

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
        y = np_utils.to_categorical(y, num_classes=4)
    return y, encoder

def prepare_data(inp_path, split_size=0.25, shuffle=False, oversize=True, target_ratio=0.8):
    x = list()
    with open(inp_path) as f:
        x_reader = csv.reader(f, delimiter=";")
        for row in x_reader:
            x.append(row)
    x = np.array(x)

    y_index = np.where(x[0,:] == "Target")[0][0]
    y = x[1:,y_index].astype(np.float32)-1 #-1 is used to scale the target column to 0-base
    x = np.delete(x, y_index, axis=1)[1:,:].astype(np.float32)

    x, _ = preprocess_data(x)

    if(oversize):
        y = y.astype(np.int32)
        count_occ = defaultdict(int)
        for i in y:
            count_occ[i] += 1
        max_occ=(0,0)
        for i in count_occ.items():
            if(i[1]>max_occ[1]):
                max_occ = i
        sampling_dict = dict()
        for i in count_occ.items():
            if(i[0]!=max_occ[0] and i[1]<round(max_occ[1]*target_ratio)):
                sampling_dict[i[0]]=round(max_occ[1]*target_ratio)
            else:
                sampling_dict[i[0]]=max_occ[1]
        sm = SMOTE(random_state = 0, sampling_strategy = sampling_dict)
        x, y = sm.fit_sample(x, y)
        y = y.astype(np.float32)
    
    if (shuffle):
       x, y = shuffle_dataset(x, y)
    
    if split_size == 1:
        x_train = x
        x_test = None
        y_train = y
        y_test = None
    else: 
        r_size = x.shape[0]
        border = math.floor(r_size*(1-split_size))
        x_train = x[:border,:]
        x_test = x[border:,:]
        y_train = y[:border]
        y_test = y[border:]

    y_train, encoder = preprocess_labels(y_train)

    return x_train, y_train, x_test, y_test, encoder

def train(x_train, y_train, x_test, y_test, n_epochs=20, verbose=1):

    dims = x_train.shape[1]
    nb_classes = y_train.shape[1]

    model = Sequential()
    model.add(Dense(256, input_shape=(dims,), activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(160, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(216, activation = "relu"))
    model.add(Dense(nb_classes, activation = "softmax"))

    model.compile(optimizer=Adam(learning_rate=0.002090922710075333, beta_1=0.9199471584216276, beta_2=0.9788631577850126),
        loss='categorical_crossentropy', metrics=['accuracy'])


    network_history = model.fit(x_train, y_train, batch_size=32, epochs=n_epochs, verbose=verbose, validation_split=0.2)
    return model, network_history

def evaluate_acc(x_train, y_train, x_test, y_test, encoder=None):
    model, _ = train(x_train, y_train, x_test, y_test, verbose=0)
    labels=model.predict_classes(x_test, batch_size=32, verbose=0)
    labels, _ = preprocess_labels(labels, encoder=encoder)
    return accuracy_score(labels, y_test)

def score(inp_path):
    n_epochs = 20
    x_train, y_train, x_test, y_test, _ = prepare_data(inp_path, split_size=0.1, shuffle=True, oversize=True, target_ratio=0.8)
    model, network_history = train(x_train, y_train, x_test, y_test, n_epochs=n_epochs)

    model.summary()
    plot_history(network_history, n_epochs)

    labels=model.predict_classes(x_test, batch_size=32, verbose=1)
    print("Accuracy: ", accuracy_score(labels, y_test))
    print(classification_report(labels, y_test))

    print('Evaluating leave-1-out-score')
    print(leave1out_cv_score(inp_path))
    

def leave1out_cv_score(inp_path):
    xs, ys, _, _, encoder = prepare_data(inp_path, split_size=1, shuffle=True, oversize=True, target_ratio=0.8)
    return leave1out_cv(xs, ys, partial(evaluate_acc, encoder=encoder), iter=100, verbose=True)
