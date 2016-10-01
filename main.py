import numpy as np
from util.preprocess import preprocess

pp = preprocess()

# Raw data
#X_train, y_train, X_test = pp.get_data_raw()

# No numbers, but otherwise identical to raw including string categories
#X_train, y_train, X_test = pp.get_data_no_numbers()

# With no numbers and stop words removed; basically just words with spaces in between
#X_train, y_train, X_test = pp.get_data_no_stop_words()

# All of the above + processed into bag of words
# Output is a sparse matrix
# Set the max_features argument if you want to change the limit at which it cuts off (default: 5000)
# Bag of words will provide the MAX_FEATURES most common words
# Even with 5000 features, one abstract might only have 70-100 non-zero features (words)
# Execution time is ~25 sec on my laptop
X_train, y_train, X_test = pp.get_bagofwords()



# Do stuff with data...

