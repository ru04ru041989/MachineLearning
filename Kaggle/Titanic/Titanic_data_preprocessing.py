# Titanic pre processing

# importing data analysis 
import numpy as np
import pandas as pd

from sklearn import preprocessing

###########################################################################################

# loading data
train_df = pd.read_csv('raw_data/train.csv')
test_df = pd.read_csv('raw_data/test.csv')
test_id = test_df['PassengerId']

###########################################################

'''
feature: 
    keep: Survived, Pclass, Sex, Age, Fare, Embarked
    add: title (from Name), FamilySize (from SibSp, Parch), IsAlone (from FamilySize)
    del: Name, Ticket, Cabin

Na value: 
    age: fill with mean
    fare: fill with meidan
    Embarked: fill with the most freq

transfor:
    normalized: Age, Fare, FamilySize 
'''

def add_feature(df):
    df['Title'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand = False)    
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
     'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs') 
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1 # to count self
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    return df

def deal_na(df):
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    freq_port = df['Embarked'].dropna().mode()[0]  # get the most common occurance
    df['Embarked'] = df['Embarked'].fillna(freq_port)
    return df

def data_drop(df, feature_drop):
    new_df = df.copy()
    new_df = new_df.drop(feature_drop, axis = 1)
    return new_df

def category_trans(df):
    # age
    result = df.copy()
    bins = (min(df['Age'])-1, 5, 12, 18, 25, 35, 60,  max(df['Age']))
    group_names = ['Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    age_type = pd.cut(df['Age'], bins, labels = group_names)
    result['Age'] = age_type
    
    # fare
    bins = (min(df['Fare'])-1, 8, 15, 31,  max(df['Fare']))
    group_names = ['1_quartile', '2_quartile', '3_quartile', '4_quartile']
    fare_type = pd.cut(df['Fare'], bins, labels = group_names)
    result['Fare'] = fare_type
    
    # famile size
    bins = (min(df['FamilySize'])-1,1.5, 4, max(df['FamilySize']))
    group_names = ['Single', 'Small', 'Large']
    family_type = pd.cut(df['FamilySize'], bins, labels = group_names)
    result['FamilySize'] = family_type    
    return result

def data_encode(df, features):
    #features = ['Sex','Age', 'Fare', 'Embarked', 'Title', 'FamilySize']
    result = df.copy()
    df_temp = df[features].copy()
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_temp[feature])
        result[feature] = le.transform(result[feature])
    return result 

def minMax_norm(df, feature_name, feature_new_name = 0):
    result = df.copy()
    if feature_new_name == 0:
        feature_new_name = feature_name
    for i in range(len(feature_name)):
        col_name = feature_name[i]
        col_new_name = feature_new_name[i]
        max_value = df[col_name].max()
        min_value = df[col_name].min()
        result[col_new_name] = (df[col_name] - min_value) / (max_value - min_value)
    return result

def mean_norm(df, feature_name, feature_new_name = 0):
    result = df.copy()
    if feature_new_name == 0:
        feature_new_name = feature_name
    for i in range(len(feature_name)):
        col_name = feature_name[i]
        col_new_name = feature_new_name[i]        
        mean_value = df[col_name].mean()
        std_value = df[col_name].std()
        result[col_new_name] = (df[col_name] - mean_value) / std_value
    return result    

######################################################
all_df = train_df.append(test_df, sort= False)

all_df = add_feature(all_df)
all_df = deal_na(all_df)
all_df = data_drop(all_df, ['Name','Ticket','Cabin'])

#############################
'''
normalize for age and fare, familysize
'''
# mean normalized
all_df_mean_norm = mean_norm(all_df, ['Age','Fare','FamilySize'])
features = ['Sex', 'Embarked', 'Title']
all_df_mean_norm = data_encode(all_df_mean_norm, features)

# min-max normalized
all_df_minMax = minMax_norm(all_df, ['Age','Fare','FamilySize'])
features = ['Sex', 'Embarked', 'Title']
all_df_minMax = data_encode(all_df_minMax, features)
'''
category for age, fare, familysize
'''
all_df_category = category_trans(all_df)
features = ['Sex','Age', 'Fare', 'Embarked', 'Title', 'FamilySize']
all_df_category = data_encode(all_df_category, features)

############################
# output processed data
all_df.to_csv('pre_processed_data/all_df.csv', index = False)
all_df_mean_norm.to_csv('pre_processed_data/all_df_mean_norm.csv', index = False)
all_df_minMax.to_csv('pre_processed_data/all_df_minMax.csv', index = False)
all_df_category.to_csv('pre_processed_data/all_df_category.csv', index = False)



###### test
def diff_pd(df1, df2):
    """Identify differences between two pandas DataFrames"""
    assert (df1.columns == df2.columns).all(), \
        "DataFrame column names are different"
    if any(df1.dtypes != df2.dtypes):
        "Data Types are different, trying to convert"
        df2 = df2.astype(df1.dtypes)
    if df1.equals(df2):
        return None
    else:
        # need to account for np.nan != np.nan returning True
        diff_mask = (df1 != df2) & ~(df1.isnull() & df2.isnull())
        ne_stacked = diff_mask.stack()
        changed = ne_stacked[ne_stacked]
        changed.index.names = ['id', 'col']
        difference_locations = np.where(diff_mask)
        changed_from = df1.values[difference_locations]
        changed_to = df2.values[difference_locations]
        return pd.DataFrame({'from': changed_from, 'to': changed_to},
                            index=changed.index)

temp_df = pd.read_csv('pre_processed_data/all_df_category.csv')
        
diff_pd(all_df_category.reset_index(drop = True), temp_df.reset_index(drop = True))