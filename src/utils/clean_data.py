import numpy as np

cols_deleted = ['Id', 'r4h3', 'r4m3', 'r4t1', 'r4t2', 'r4t3', 'rez_esc', 'elimbasu5', 'idhogar', 'hogar_total', 'mobilephone']
cols_parsed = ['dependency', 'edjefe', 'edjefa']

def clean_data(raw):
    cleaned_data = raw.copy()

    #mean-filling variable: v2a1, Monthly rent payment
    j, = np.where(cleaned_data[0,:] == "v2a1")[0]
    mean = np.around(np.mean([float(x) for x in cleaned_data[1:,j] if x!=""]), decimals=1)
    for i in range(1, cleaned_data.shape[0]):
        if(cleaned_data[i,j] == ""):
            cleaned_data[i,j] = str(mean)
    #0-filling for column: v18q1, number of tablets household owns
    j, = np.where(cleaned_data[0,:] == "v18q1")[0]
    for i in range(1, cleaned_data.shape[0]):
        if(cleaned_data[i,j] == ""):
            cleaned_data[i,j] = "0"
    
    # fixing yes,no -> 1,0
    for col in cols_parsed:
        j, = np.where(cleaned_data[0,:] == col)[0]
        for i in range(1, cleaned_data.shape[0]):
            if(cleaned_data[i,j] == "yes"):
                cleaned_data[i,j] = 1
            elif(cleaned_data[i,j] == "no"):
                cleaned_data[i,j] = 0

    # Removing columns
    for col in cols_deleted:
        j, = np.where(cleaned_data[0,:] == col)[0]
        cleaned_data = np.delete(cleaned_data, j, axis=1)

    # Removing rows
    # removing rows with empty variable: meaneduc / SQBmeaned
    j, = np.where(cleaned_data[0,:] == "meaneduc")[0]
    cleaned_data_rows_dim = cleaned_data.shape[0]
    i = 0
    while(i<cleaned_data_rows_dim):
        if(cleaned_data[i,j] == ""):
            cleaned_data = np.delete(cleaned_data, i, axis=0)
            cleaned_data_rows_dim -= 1
        else:
            i += 1
    return cleaned_data