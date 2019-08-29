# importing data analysis 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
import scipy.stats as st
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#####################################################################

def plot_scatter(df, x, y = 'SalePrice'):
    data = pd.concat([df[x],df[y]], axis = 1)
    data.plot.scatter(x = x, y = y, ylim = (0, 1.2* max(df[y])))
    
def plot_box(df,x,y = 'SalePrice'):
    data = pd.concat([df[x],df[y]], axis = 1)
    sns.boxplot(x = x, y = y, data = data)
    x = plt.xticks(rotation = 90)

def plot_missing(df):
    missing = df.isnull().sum()
    missing = missing[missing>0]
    missing.sort_values(inplace = True)
    missing.plot.bar()

###########################################################################################

# loading data
train_df = pd.read_csv('raw_data/train.csv')
test_df = pd.read_csv('raw_data/test.csv')

########################################################

# identify quant and qualitative data
quantitative = [f for f in train_df.columns if train_df.dtypes[f] != 'object']
qualitative  = [f for f in train_df.columns if train_df.dtypes[f] == 'object']

# check normalite 
non_norm_list = []
for item in quantitative:
    if stats.shapiro(train_df[item])[1] < 0.01:
        non_norm_list.append(item)

## Categorical data
# fill na with missing
for c in qualitative:
    train_df[c] = train_df[c].astype('category')
    if train_df[c].isnull().any():
        train_df[c] = train_df[c].cat.add_categories(['MISSING'])
        train_df[c] = train_df[c].fillna('MISSING')

        
############        
        
        
ordering = pd.DataFrame()
ordering['val'] = train_df['MSZoning'].unique()
ordering.index = ordering.val
ordering['spmean'] = train_df[['MSZoning', 'SalePrice']].groupby('MSZoning').mean()['SalePrice']
ordering = ordering.sort_values('spmean')
ordering['ordering'] = range(1, ordering.shape[0]+1)
ordering = ordering['ordering'].to_dict()
        
for cat, o in ordering.items():
    train_df.loc[train_df['MSZoning'] == cat, 'MSZoning'+'_E'] = o
        
        
#standardizing data
saleprice_scaled = StandardScaler().fit(train_df['SalePrice'][:,np.newaxis])
trans_price = saleprice_scaled.transform(train_df['SalePrice'][:,np.newaxis])
trans_price = sorted(trans_price)

a = np.reshape(trans_price, len(trans_price))

print(trans_price[:10])     