from scipy import spatial

dataSetI = [3018, 856, 632, 429]
dataSetII = [2973, 741, 577, 326]
result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
print(result)