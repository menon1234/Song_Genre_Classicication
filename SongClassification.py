import keras
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import spacy
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

##Check if GPU is availablee
tf.test.gpu_device_name()
## Import necessary Documents

import pandas as pd

# import readr
from sklearn.model_selection import cross_val_score, StratifiedKFold

##Import the dataframe
