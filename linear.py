import tensorflow as tf
import tensorflow.feature_column as fc
import os
import sys
import matplotlib.pyplot as plt
# from IPython.display import clear_output

tf.enable_eager_execution()
sys.path.append(r'D:\tensorflow_examples\models')
from official.wide_deep import census_dataset
from official.wide_deep import census_main
census_dataset.download("/tmp/census_data/")
train_file = "/tmp/census_data/adult.data"
test_file = "/tmp/census_data/adult.test"
import pandas
train_df = pandas.read_csv(train_file, header = None, names = census_dataset._CSV_COLUMNS)
test_df = pandas.read_csv(test_file, header = None, names = census_dataset._CSV_COLUMNS)
train_df.head()
train_df.head()
