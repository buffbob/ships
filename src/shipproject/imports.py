import warnings
warnings.filterwarnings( "ignore", module = "matplotlib\..*" )
warnings.filterwarnings( "ignore", module = "tensorflow" )


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import json
from tensorflow.keras import layers
from tensorflow.keras import Sequential

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix





print ("imports added!");