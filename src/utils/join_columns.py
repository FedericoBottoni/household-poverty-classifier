import numpy as np

def join_columns(raw_train, col_names, col_map, new_label, order, new_oth_names={}):
    #creating new int column
    raw_train = np.concatenate((raw_train, np.zeros((raw_train.shape[0],1), dtype=np.int32).astype('str')), axis=1)
    raw_train[0,-1] = new_label
    #saving index for later use
    new_int_col_index = raw_train.shape[1]-1
    #creating optional boolen columns
    new_oth_col_index = dict()
    for k in new_oth_names.keys():
        raw_train = np.concatenate((raw_train, np.zeros((raw_train.shape[0],1), dtype=np.int32).astype('str')), axis=1)
        raw_train[0,-1] = new_oth_names[k]
        #saving index for later use
        new_oth_col_index[k] = raw_train.shape[1]-1
    #building index list of features columns
    col_group = np.zeros(len(col_names), dtype=np.int32)
    for i in range(len(col_names)):
        col_group[i], = np.where(raw_train[0,:] == col_names[i])[0]
    #updating raw_train
    for i in range(1, raw_train.shape[0]):
        #building categorical row
        row = [raw_train[i,s] for s in col_group]
        row = np.array(row)
        #finding selected cell content and type
        index, = np.where(row == "1")[0]
        index_type = col_map[index]
        #if type is not categorical
        if(index_type != "c"):
            #update new boolean column
            raw_train[i,new_oth_col_index[index_type]] = "1"
        #if categorial
        else:
            #update new int column
            raw_train[i,new_int_col_index] = str(order[index])
    #removing old columns
    #sorting index list of features columns in order to simplify the iterative removing process
    col_group = np.sort(col_group)
    col_group_cols_dim = col_group.shape[0]
    i = 0
    while(i<col_group_cols_dim):
        raw_train = np.delete(raw_train, col_group[i], axis=1)
        col_group -= 1
        i += 1
    return raw_train