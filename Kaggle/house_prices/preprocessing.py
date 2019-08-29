# importing data analysis 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from scipy import stats
import warnings
warnings.filterwarnings('ignore')


###########################################################################################

# loading data
train_df = pd.read_csv('raw_data/train.csv')
test_df = pd.read_csv('raw_data/test.csv')

