
from collections import OrderedDict
from pyGPGO.covfunc import squaredExponential
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.surrogates.RandomForest import RandomForest
from pyGPGO.GPGO import GPGO

def gproc(evaluate_network, params, acquisition_function = 'ExpectedImprovement'):
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
    return res[0], res[1]


def rforest(evaluate_network, params, acquisition_function = 'ExpectedImprovement'):
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

def sample_hps(evaluate_network, params, acquisition_function):
    best_params, best_acc = None, 0
    surrogates = [gproc, rforest]
    for surrogate in surrogates
        next_params, next_acc = surrogate(evaluate_network, params, acquisition_function)
        if best_acc < next_acc:
            best_acc = next_acc
            best_params = next_params
    print('Sampled', best_params, 'with score', best_acc)
    return best_params
    