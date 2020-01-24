import math
import numpy as np
import csv as csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from model_fairness import model_fairness

def prepare_data_fairness(inp_path, fields_list=[], split_size=0.25, shuffle=False):
    
    x = list()
    with open(inp_path) as f:
        x_reader = csv.reader(f, delimiter=";")
        for row in x_reader:
            x.append(row)
    x = np.array(x)
    data_dictionary={}
    index_list=list()
    for target in fields_list:
        index_list.append(np.where(x[0,:] == str(target))[0][0])
        data_dictionary[target]={"count":0, "extreme_poverty_count":0, "moderate_povery_count":0, "vulnerable_households_count":0, "non_vulnerable_households_count":0}

    y_index = np.where(x[0,:] == "Target")[0][0]
    x=x[1:,:]
    if (shuffle):
        x = shuffle(x[:,])

    data_distribution=list()
    for element in index_list:
        data_distribution.append(0)

    data_distribution= np.array(data_distribution)
    for  row in x:
        i=0
        for index in index_list:
            if(row[index.astype(int)].astype(int)==1):
                data_distribution[i]+=1
                data_dictionary[fields_list[i]]["count"]+=1
                if(row[y_index].astype(int)==1):
                    data_dictionary[fields_list[i]]["extreme_poverty_count"]+=1
                else:
                    if(row[y_index].astype(int)==2):
                        data_dictionary[fields_list[i]]["moderate_povery_count"]+=1
                    else:
                        if(row[y_index].astype(int)==3):
                            data_dictionary[fields_list[i]]["vulnerable_households_count"]+=1
                        else:
                            if(row[y_index].astype(int)==4):
                                data_dictionary[fields_list[i]]["non_vulnerable_households_count"]+=1
            i+=1
    return data_dictionary

def check_flist(inp_path):
    print("Dataset fairness grouping by fields")
    fields_list=list()
    fields_list.append("male")
    fields_list.append("female")
    #fields_list.append("instlevel1")
    #fields_list.append("instlevel2")
    #fields_list.append("instlevel3")
    #fields_list.append("instlevel4")
    #fields_list.append("instlevel5")
    #fields_list.append("instlevel6")
    #fields_list.append("instlevel7")
    #fields_list.append("instlevel8")
    #fields_list.append("instlevel9")
    #print(fields_list)
    data_dictionary = prepare_data_fairness(inp_path, fields_list)
    print("----------------------")
    for data in data_dictionary:
        print(data)
        print(data_dictionary[data])
        print("----------------------")
    model_fairness(inp_path)
