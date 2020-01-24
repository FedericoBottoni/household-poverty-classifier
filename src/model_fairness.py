import math
import numpy as np
import csv as csv
from functools import partial
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout
from train import preprocess_data, preprocess_labels, train

def prepare_data(inp_path, split_size=0.25, target_ratio=0.8):
    x = list()
    with open(inp_path) as f:
        x_reader = csv.reader(f, delimiter=";")
        for row in x_reader:
            x.append(row)
    x = np.array(x)

    y_index = np.where(x[0,:] == "Target")[0][0]
    male_index=np.where(x[0,:] == str("male"))[0][0]
    female_index=np.where(x[0,:] == str("female"))[0][0]
    #y = x[1:,y_index].astype(np.float32)-1 #-1 is used to scale the target column to 0-base
    y = x[1:,y_index].astype(np.float32)-1

    r_size = x.shape[0]
    border = math.floor(r_size*(1-split_size))

    x_test_normal = x[border:,:]
    #male and female count

    #swap male and female column, save it and restore the orginal dataset
    male_column=x_test_normal[:,male_index]

    female_column=x_test_normal[:,female_index]
    x_test_fair=x_test_normal
    x_test_fair[:,male_index] = female_column
    x_test_fair[:,female_index] = male_column

    target_list=list()
    c=0
    for target in x_test_normal:
        target_list.append({"index":c, "gender":int(target[np.where(x[0,:] == str("male"))[0][0]]), "true_target":int(target[y_index]), "predicted_targed":0, "fair_target":0})
        c+=1
        
    x_test_fair = np.delete(x_test_fair, y_index, axis=1)[1:,:].astype(np.float32)
    x_test_fair, _ = preprocess_data(x_test_fair)

    x = np.delete(x, y_index, axis=1)[1:,:].astype(np.float32)
    x, _ = preprocess_data(x)

    x_train = x[:border,:]
    
    y_train = y[:border]
    y_train, encoder = preprocess_labels(y_train)
    
    x_test = x[border:,:]
   
    y_test = y[border:]

    return x_train, y_train, x_test, y_test, encoder, target_list, x_test_fair


def model_fairness(inp_path):
    print("Model fairness after the gender swap")
    x_train, y_train, x_test, y_test, _, target_list, x_test_fair = prepare_data(inp_path)

    model, _ = train(x_train, y_train, x_test, y_test, verbose=0)
    labels=model.predict_classes(x_test, batch_size=32, verbose=0)
    c=0
    for l in labels:
        target_list[c]["predicted_targed"]=l
        c+=1

    labels=model.predict_classes(x_test_fair, batch_size=32, verbose=0)
    c=0
    for l in labels:
        target_list[c]["fair_target"]=l
        c+=1


    male_up=0
    male_down=0
    male_fair=0
    female_up=0
    female_down=0
    female_fair=0

    for e in target_list:
        if(e["gender"]==1):
            if(e["predicted_targed"]<e["fair_target"]):
                male_up+=1
            else:
                if(e["predicted_targed"]>e["fair_target"]):
                    male_down+=1
                else:
                    male_fair+=1
        else:
            if(e["predicted_targed"]<e["fair_target"]):
                female_up+=1
            else:
                if(e["predicted_targed"]>e["fair_target"]):
                    female_down+=1
                else:
                    female_fair+=1
    print("--------------------")
    print("Male who increased:")
    print(male_up)
    print("Male who decreased:")
    print(male_down)
    print("Male fair")
    print(male_fair)
    print("--------------------")
    print("Female who increased:")
    print(female_up)
    print("Female who decreased:")
    print(female_down)
    print("Female fair")
    print(female_fair)