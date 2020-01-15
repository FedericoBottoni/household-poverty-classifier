import numpy as np
from sklearn.model_selection import KFold

def score_cv(xs, ys, model_evaluate, verbose=False):
    kfolds = 10
    kf = KFold(n_splits=kfolds)
    accs = [None] * kfolds
    i=0
    for train_index, test_index in kf.split(xs):
        x_train, x_test = xs[train_index], xs[test_index]
        y_train, y_test = ys[train_index], ys[test_index]
        acc = model_evaluate(x_train, y_train, x_test, y_test)
        np.append(accs, acc)
        accs[i] = acc
        i=i+1
        if verbose:
            print('CV:', i, '/', kfolds, ' acc', acc)
    mean_accs = np.array(accs).mean()
    if verbose:
        print('CV mean acc:', mean_accs)
    return 
    
def leave1out_cv(xs, ys, model_evaluate, iter=100, verbose=False):
    accs = [None] * iter
    nouts = np.random.randint(0,len(ys),iter)
    for i in range(1, iter):
        x_train = xs.copy().tolist()
        y_train = ys.copy().tolist()
        x_test = x_train.pop(nouts[i])
        y_test = y_train.pop(nouts[i])

        acc = model_evaluate(np.array(x_train), np.array(y_train), np.array([x_test]), np.array([y_test]))
        np.append(accs, acc)
        accs[i] = acc
        if verbose:
            print('CV:', i, '/', iter, ' acc', acc)
        i=i+1
    mean_accs = np.array(accs).mean()
    if verbose:
        print('CV mean acc:', mean_accs)
    return mean_accs