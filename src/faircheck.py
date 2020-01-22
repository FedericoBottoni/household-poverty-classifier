from fairsearchdeltr import Deltr
# import other helper libraries
import pandas as pd
from io import StringIO

# load some train data (this is just a sample - more is better)
train_data_raw = "./data/train.csv"
train_data = pd.read_csv(train_data_raw, sep=';')
print(train_data)
# setup the DELTR object
protected_feature = "female" # column name of the protected attribute (index after query and document id)
gamma = 1 # value of the gamma parameter
number_of_iterations = 1000 # number of iterations the training should run
standardize = True # let's apply standardization to the features

# create the Deltr object
dtr = Deltr(protected_feature, gamma, number_of_iterations, standardize=standardize)

# train the model
dtr.train(train_data)

# your run should have approximately same results  

