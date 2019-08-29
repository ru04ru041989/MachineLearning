# digit recognizer nn 

## import module
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from keras.wrappers.scikit_learn import KerasClassifier

import matplotlib.pyplot as plt, matplotlib.image as mpimg

import pandas as pd
import numpy as np


## load data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv') 

## extract X and y
X_train = train_df.drop(['label'], axis = 1).astype('float32') # all pixel values
y_train = pd.DataFrame(train_df['label']) # only labels i.e targets digits

## Data Augmentation

# data_set to augmentation
#   select n time of the picture for each number
def get_aug_data_set(x_df, y_df, n_num):
    count = np.zeros(10)
    X_aug = pd.DataFrame()
    y_aug = []
    
    gen =ImageDataGenerator(rotation_range=4, width_shift_range=0.05,
                            height_shift_range=0.05, zoom_range=0.05)
    
    for i in range(x_df.shape[0]):
        label = y_df['label'][i]
        if count[label] < n_num:
            a_img = x_df.iloc[i].values
            a_img = a_img.reshape(1,28,28,1)
            a = gen.flow(a_img)
            img_ls = a[0][0].reshape(1,784)
        
            X_aug = X_aug.append(pd.DataFrame(img_ls))
            y_aug.append(y_df['label'][i])
            count[label] += 1
    X_aug.columns = x_df.columns
    y_aug = pd.DataFrame(y_aug)
    y_aug.columns = y_df.columns    
    return X_aug, y_aug
###
is_aug = 0
if is_aug:
    x_aug, y_aug = get_aug_data_set(X_train, y_train, 20000)


# save the aug data
x_aug.to_csv('x_aug.csv', index = False)
y_aug.to_csv('y_aug.csv', index = False)


## function to display img
def show_ditit_ls(i):
    img = i.reshape((28,28))
    img = img.reshape((28,28))
    plt.imshow(img,cmap='gray')

def show_digit_df(i, x_df, y_df):
    img = x_df.iloc[i].values
    img = img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(y_df.iloc[i].values)

#########################################################################
## function for NN

def run_nn_kfold(model, x_all, y_all, n_folds = 10, print_each = False, print_head = 3):
    kf = KFold(n_splits =  n_folds)
    outcomes = []
    fold = 0
    for train_index, test_index in kf.split(x_all):
        fold += 1
        x_train, x_test = x_all.values[train_index], x_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        scores = model.evaluate(x_train, y_train)
        outcomes.append(scores[1])
        if print_each:
            print("Fold {} accuracy: {}".format(fold, scores[1]))
    mean_outcome = np.mean(outcomes)
    #print(outcomes[:print_head])
    #print("Mean Accuracy: {}".format(mean_outcome))
    return mean_outcome
## 
    
X_all = X_train.append(x_aug)
y_all = y_train.append(y_aug)

# preprocessing the digit images
mM_scaler = MinMaxScaler()
x_mMscaler = pd.DataFrame(mM_scaler.fit_transform(X_all))

onehot_encoder = OneHotEncoder(sparse=False)
y_onehot = pd.DataFrame(onehot_encoder.fit_transform(y_all))



input_dim = x_mMscaler.shape[1]
nb_classes = y_onehot.shape[1]

##
# NN model- Liner Model

# creat model
model = Sequential()
model.add(Dense(128, input_dim = input_dim, activation='relu'))
model.add(Dropout(0.15))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.15))
model.add(Dense(nb_classes, activation = 'softmax'))

# complie
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001), metrics = ['accuracy'])

# Fit the model
model.fit(x_mMscaler, y_onehot, epochs = 100, batch_size = 10, verbose=2)

# evaluate the model
evalu_nn_model = (model, model.evaluate(x_mMscaler, y_onehot), 
                  run_nn_kfold(model, x_mMscaler, y_onehot))

print(evalu_nn_model[1])
print(evalu_nn_model[2])

# prediction
prediction_onehot = model.predict(test_df)
prediction = np.dot(np.array(prediction_onehot), np.array(range(10))).astype('int32')
################################################################################

# tune nn

'''
Tune the Training Optimization Algorithm
'''

# Function to create model, required for KerasClassifier
def create_model(optimizer='adam'):
    # create model
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.15))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.15))
    model.add(Dense(nb_classes, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
model = create_model()

# Fit the model
model.fit(x_mMscaler, y_onehot, epochs = 100, batch_size = 10, verbose=2)

# evaluate the model
evalu_nn_model = (model, model.evaluate(x_mMscaler, y_onehot), 
                  run_nn_kfold(model, x_mMscaler, y_onehot))

print(evalu_nn_model[1])
print(evalu_nn_model[2])

# prediction
prediction_onehot = model.predict(test_df)
prediction = np.dot(np.array(prediction_onehot), np.array(range(10))).astype('int32')


#########################################
# output
is_output = 1
if is_output:
    output = pd.DataFrame()
    output['ImageId'] = list(range(1,len(prediction)+1))
    output['Label'] = prediction

    output.to_csv('prediction/nn_3.csv', index = False)
    
    
############
# compare output

output1 = pd.read_csv('prediction/nn_3.csv')
output2 = pd.read_csv('prediction/nn_2.csv')

a = output1['Label']
b = output2['Label']
c = b-a