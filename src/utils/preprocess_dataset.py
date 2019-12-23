import numpy as np
import csv as csv
from clean_data import clean_data
from join_columns import join_columns

raw_train = list()
with open("../../data/raw_train.csv", encoding="utf-8") as f:
    raw_train_reader = csv.reader(f, delimiter=",")
    for row in raw_train_reader:
        raw_train.append(row)
raw_train = np.array(raw_train)

raw_train = clean_data(raw_train)

raw_train = join_columns(raw_train, ["sanitario1", "sanitario2", "sanitario3", "sanitario5", "sanitario6"], ["c","c","c","c","o1"], "sanitario", [1,2,3,4], {"o1":"sanioth"})
raw_train = join_columns(raw_train, ["energcocinar1", "energcocinar2", "energcocinar3", "energcocinar4"], ["c","c","c","c"], "energcocinar", [1,4,2,3])
raw_train = join_columns(raw_train, ["elimbasu1", "elimbasu2", "elimbasu3", "elimbasu4", "elimbasu6"], ["c","c","c","c","o1"], "elimbasu", [4,3,2,1], {"o1":"elimoth"})
raw_train = np.delete(raw_train, np.where(raw_train[0,:] == "elimbasu5")[0][0], axis=1)
raw_train = join_columns(raw_train, ["epared1", "epared2", "epared3"], ["c","c","c"], "epared", [1,2,3])
raw_train = join_columns(raw_train, ["etecho1", "etecho2", "etecho3"], ["c","c","c"], "etecho", [1,2,3])
raw_train = join_columns(raw_train, ["eviv1", "eviv2", "eviv3"], ["c","c","c"], "eviv", [1,2,3])
raw_train = join_columns(raw_train, ["female", "male"], ["c","c"], "gender", [0,1])
raw_train = join_columns(raw_train, ["parentesco1", "parentesco2", "parentesco3", "parentesco4", "parentesco5", "parentesco6", "parentesco7", "parentesco8", "parentesco9", "parentesco10", "parentesco11", "parentesco12"], ["c","c","c","c","c","c","c","c","c","c","c","c"], "parentesco", [1,2,3,4,5,6,7,8,9,10,11,12])
raw_train = join_columns(raw_train, ["instlevel1", "instlevel2", "instlevel3", "instlevel4", "instlevel5", "instlevel6", "instlevel7", "instlevel8", "instlevel9"], ["c","c","c","c","c","c","c","c","c"], "instlevel", [1,2,3,4,5,6,7,8,9])
raw_train = join_columns(raw_train, ["tipovivi1", "tipovivi2", "tipovivi3", "tipovivi4", "tipovivi5"], ["c","c","c","c","o1"], "tipovivi", [1,2,3,4], {"o1":"tipooth"})
raw_train = join_columns(raw_train, ["area2", "area1"], ["c","c"], "area", [0,1])

#saving new dataset
np.savetxt('../../data/train.csv', raw_train, delimiter=';', fmt='%s')