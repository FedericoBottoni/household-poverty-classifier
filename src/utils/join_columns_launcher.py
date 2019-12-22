import numpy as np
import csv as csv
from join_columns import join_columns

raw_train = list()
with open("../../data/raw_train_cleaned.csv", encoding="utf-8") as f:
    raw_train_reader = csv.reader(f, delimiter=";")
    for row in raw_train_reader:
        raw_train.append(row)
raw_train = np.array(raw_train)

raw_train = join_columns(raw_train, ["sanitario1", "sanitario2", "sanitario3", "sanitario5", "sanitario6"], ["c","c","c","c","o1"], "sanitario", [1,2,3,4], {"o1":"sanioth"})
raw_train = join_columns(raw_train, ["energcocinar1", "energcocinar2", "energcocinar3", "energcocinar4"], ["c","c","c","c"], "energcocinar", [1,4,2,3])

#saving new dataset
np.savetxt('../../data/train.csv', raw_train, delimiter=';', fmt='%s')