import numpy as np
import csv as csv

raw_train = list()
with open("../../data/raw_train.csv", encoding="utf-8") as f:
    raw_train_reader = csv.reader(f, delimiter=",")
    for row in raw_train_reader:
        raw_train.append(row)
raw_train = np.array(raw_train)
empty_rows_count = list()
for j in range(raw_train.shape[1]):
    count = 0
    for i in range(1, raw_train.shape[0]):
        if(raw_train[i,j] == ""):
            count += 1
    empty_rows_count.append(count)
empty_rows_count = np.array(np.array(empty_rows_count))
print([x for x in empty_rows_count if x!=0])
np.savetxt('../../data/empty_rows_count.csv', np.concatenate((np.expand_dims(raw_train[0,:], axis=0), np.expand_dims(empty_rows_count, axis=0)), axis=0), delimiter=';', fmt='%s')