import math
import numpy as np
import csv as csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation


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

def prepare_data_fairness(inp_path, split_size=0.25, shuffle=False):
    x = list()
    with open(inp_path) as f:
        x_reader = csv.reader(f, delimiter=";")
        for row in x_reader:
            x.append(row)
    x = np.array(x)
    #print(type(x))
    #print(x[0,66:68]) 
    y_index = np.where(x[0,:] == "Target")[0][0]
    print(y_index)
    x=x[1:,:]
    if (shuffle):
        x = shuffle(x[:,]) #Remove the first row and shuffle 
    
    genders=x[:,66:68]
    y_test_male=list()
    y_test_female=list()
    male1_count=0
    male2_count=0
    male3_count=0
    male4_count=0
    female1_count=0
    female2_count=0
    female3_count=0
    female4_count=0
    c=0
    for  row in genders:
        
        if(row[1].astype(int)==0):
            y_test_male.append(c)
            #print(x[c,131])
            if(x[c,131].astype(int)==1):
                male1_count+=1
            else:
                if(x[c,131].astype(int)==2):
                    male2_count+=1
                else:
                    if(x[c,131].astype(int)==3):
                        male3_count+=1
                    else:
                        male4_count+=1
        else:
            y_test_female.append(c)
            if(x[c,131].astype(int)==1):
                female1_count+=1
            else:
                if(x[c,131].astype(int)==2):
                    female2_count+=1
                else:
                    if(x[c,131].astype(int)==3):
                        female3_count+=1
                    else:
                        female4_count+=1 
        c+=1
    y_test_female=np.array(y_test_female)
    y_test_male=np.array(y_test_male)
    print("Male stats")
    print("Male count:")
    print(y_test_male.size)
    print("extreme poverty male count:")
    print(male1_count)
    print("moderate poverty male count:")    
    print(male2_count)
    print("vulnerable households male count:")
    print(male3_count)
    print(" non vulnerable households male count:")
    print(male4_count)    
    print("--------------------------")
    print("Male stats")
    print("Male count:")
    print(y_test_female.size)
    print("extreme poverty female count:")
    print(female1_count)
    print("moderate poverty female count:") 
    print(female2_count)
    print("vulnerable households female count:")
    print(female3_count)
    print(" non vulnerable households female count:")
    print(female4_count)
    #print(x[y_test_male[0]])

    y = x[1:,y_index].astype(np.float32)-1 #-1 is used to scale the target column to 0-base
    x = np.delete(x, y_index, axis=1)[1:,:].astype(np.float32)
    x, _ = preprocess_data(x)
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


prepare_data_fairness("./data/train.csv")