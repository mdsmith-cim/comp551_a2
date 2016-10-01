import numpy as np
from util.preprocess import  preprocess

pp = preprocess()
X_train, y_train, X_test = pp.get_data_process(max_features=5000)

# Do stuff with data...

