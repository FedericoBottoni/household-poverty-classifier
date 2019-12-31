import numpy as np

def add_int(raw, num):
    for i in range(1, raw.shape[0]):
        for j in range(raw.shape[1]):
            try:
                s = str(raw[i,j])
                if(s.index('.') == 0):
                    f = float(str(num) + s)
                    raw[i,j] = str(f)
            except:
                pass
    return raw

def cut_decimals(raw, decimals):
    for i in range(1, raw.shape[0]):
        for j in range(raw.shape[1]):
            try:
                f = float(raw[i,j])
                s = str(f)
                if '.' in s:
                    if(len(s) - s.index('.') - 1 > decimals):
                        raw[i,j] = s[:s.index('.') + 1 + decimals]
            except:
                pass
    return raw