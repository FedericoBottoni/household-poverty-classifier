import array
import numpy as np
from collections import OrderedDict
from functools import partial
from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.surrogates.RandomForest import RandomForest
from pyGPGO.GPGO import GPGO
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from train import prepare_data, preprocess_labels
from score_helper import score_cv

def gproc(evaluate_network, params, acquisition_function='ExpectedImprovement'):
    print('init Gaussian process')
    sexp = squaredExponential()
    gp = GaussianProcess(sexp)
    acq = Acquisition(mode=acquisition_function)

    gpgo = GPGO(gp, acq, evaluate_network, params)
    try:
        gpgo.run(init_evals=5, max_iter=20)
        res = gpgo.getResult()
    except Exception as err:
        print("Error handled: ", err)
        res = [None, 0]
        raise err
    return res[0], res[1]


def rforest(evaluate_network, params, acquisition_function='ExpectedImprovement'):
    print('init Random forest')
    rf = RandomForest()
    acq = Acquisition(mode=acquisition_function)

    rf1 = GPGO(rf, acq, evaluate_network, params)    
    try:
        rf1.run(init_evals=5, max_iter=20)
        res = rf1.getResult()
    except Exception as err:
        print("Error handled: ", err)
        res = [None, 0]
    return res[0], res[1]
    
def evaluate_nn(x_train, y_train, x_test, y_test, learning_rate, beta_1, beta_2, encoder):
    dims = x_train.shape[1]
    nb_classes = y_train.shape[1]

    model = Sequential()
    model.add(Dense(256, input_shape=(dims,), activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation = "relu"))
    model.add(Dense(nb_classes, activation = "softmax"))

    model.compile(optimizer=Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2), loss='categorical_crossentropy', metrics=['accuracy'])

    n_epochs = 10
    batch_size = 32
    network_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epochs, verbose=0, validation_split=0.2)
    labels=model.predict_classes(x_test, batch_size=batch_size, verbose=0)
    processed, _ = preprocess_labels(labels, encoder=encoder)
    acc = accuracy_score(processed, y_test)
    return acc

def evaluate_cv(inp_path, learning_rate, beta_1, beta_2):
    xs, ys, _, _, encoder = prepare_data(inp_path, split_size=1, shuffle=True, oversize=True)
    return score_cv(xs, ys, partial(evaluate_nn, learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, encoder=encoder), verbose=True)

def sample_hps(input_params):
    best_params, best_acc = None, 0
    surrogates = [rforest, gproc]
    evaluate_cv_bound = partial(evaluate_cv, input_params)
    params = OrderedDict()
    params['learning_rate'] = ('cont', [0.001, 0.01])
    params['beta_1'] = ('cont', [0.8, 0.999])
    params['beta_2'] = ('cont', [0.8, 0.999])

    acquisition_function = 'ExpectedImprovement'
    for surrogate in surrogates:
        next_params, next_acc = surrogate(evaluate_cv_bound, params, acquisition_function)
        if best_acc < next_acc:
            best_acc = next_acc
            best_params = next_params
    print('Sampled', best_params, 'with score', best_acc)
    return best_params