import numpy as np
import csv as csv

raw_train = list()
with open("../../data/raw_train.csv", encoding="utf-8") as f:
    raw_train_reader = csv.reader(f, delimiter=",")
    for row in raw_train_reader:
        raw_train.append(row)
raw_train = np.array(raw_train)
#mean-filling variable: v2a1, Monthly rent payment
j, = np.where(raw_train[0,:] == "v2a1")[0]
mean = np.around(np.mean([float(x) for x in raw_train[1:,j] if x!=""]), decimals=1)
for i in range(1, raw_train.shape[0]):
    if(raw_train[i,j] == ""):
        raw_train[i,j] = str(mean)
#0-filling for column: v18q1, number of tablets household owns
j, = np.where(raw_train[0,:] == "v18q1")[0]
for i in range(1, raw_train.shape[0]):
    if(raw_train[i,j] == ""):
        raw_train[i,j] = "0"
#removing column: rez_esc, Years behind in school
j, = np.where(raw_train[0,:] == "rez_esc")[0]
raw_train = np.delete(raw_train, j, axis=1)
#removing rows with empty variable: meaneduc / SQBmeaned
j, = np.where(raw_train[0,:] == "meaneduc")[0]
raw_train_rows_dim = raw_train.shape[0]
i = 0
while(i<raw_train_rows_dim):
    if(raw_train[i,j] == ""):
        raw_train = np.delete(raw_train, i, axis=0)
        raw_train_rows_dim -= 1
    else:
        i += 1
#saving new dataset
np.savetxt('../../data/raw_train_cleaned.csv', raw_train, delimiter=';', fmt='%s')