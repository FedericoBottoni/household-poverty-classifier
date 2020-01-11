import numpy as np
import csv as csv
from clean_data import clean_data
from join_columns import join_columns
from fix_decimals import add_int, cut_decimals

def preprocess_dataset():
    preprocess_data('train', False)
    preprocess_data('test', False)
    preprocess_data('train', True)
    preprocess_data('test', True)

def preprocess_data(data_name, encode_features):
    name = data_name
    raw = list()
    with open("./data/raw_" + data_name + ".csv") as f:
        raw_reader = csv.reader(f, delimiter=",")
        for row in raw_reader:
            raw.append(row)
    raw = np.array(raw)
    raw = clean_data(raw)
    if encode_features:
        raw = join_columns(raw, ["sanitario1", "sanitario2", "sanitario3", "sanitario5", "sanitario6"], ["c","c","c","c","o1"], "sanitario", [1,2,3,4], {"o1":"sanioth"})
        raw = join_columns(raw, ["energcocinar1", "energcocinar2", "energcocinar3", "energcocinar4"], ["c","c","c","c"], "energcocinar", [1,4,2,3])
        raw = join_columns(raw, ["elimbasu1", "elimbasu2", "elimbasu3", "elimbasu4", "elimbasu6"], ["c","c","c","c","o1"], "elimbasu", [4,3,2,1], {"o1":"elimoth"})
        #raw = np.delete(raw, np.where(raw[0,:] == "elimbasu5")[0][0], axis=1) #this column has been removed inside the clean_data function since it has 0 mean and 0 variance
        raw = join_columns(raw, ["epared1", "epared2", "epared3"], ["c","c","c"], "epared", [1,2,3])
        raw = join_columns(raw, ["etecho1", "etecho2", "etecho3"], ["c","c","c"], "etecho", [1,2,3])
        raw = join_columns(raw, ["eviv1", "eviv2", "eviv3"], ["c","c","c"], "eviv", [1,2,3])
        raw = join_columns(raw, ["female", "male"], ["c","c"], "gender", [0,1])
        raw = join_columns(raw, ["parentesco1", "parentesco2", "parentesco3", "parentesco4", "parentesco5", "parentesco6", "parentesco7", "parentesco8", "parentesco9", "parentesco10", "parentesco11", "parentesco12"], ["c","c","c","c","c","c","c","c","c","c","c","c"], "parentesco", [1,2,3,4,5,6,7,8,9,10,11,12])
        raw = join_columns(raw, ["instlevel1", "instlevel2", "instlevel3", "instlevel4", "instlevel5", "instlevel6", "instlevel7", "instlevel8", "instlevel9"], ["c","c","c","c","c","c","c","c","c"], "instlevel", [1,2,3,4,5,6,7,8,9])
        raw = join_columns(raw, ["tipovivi1", "tipovivi2", "tipovivi3", "tipovivi4", "tipovivi5"], ["c","c","c","c","o1"], "tipovivi", [1,2,3,4], {"o1":"tipooth"})
        raw = join_columns(raw, ["area2", "area1"], ["c","c"], "area", [0,1])
        name = name + '_enc'
    raw = add_int(raw, 0)
    raw = cut_decimals(raw, 2)

    #saving new dataset
    print('exporting ' + name + '.csv')
    np.savetxt('./data/' + name + '.csv', raw, delimiter=';', fmt='%s')