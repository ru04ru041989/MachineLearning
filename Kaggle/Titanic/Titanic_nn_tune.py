# Titanic nn tune

# importing data analysis 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# Modelling Helpers
from sklearn import preprocessing

# import module
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import KFold

# deep learning
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier

from keras.layers import Dropout
from keras.constraints import maxnorm

#########################################
def data_encode(df, features):
    #features = ['Sex','Age', 'Fare', 'Embarked', 'Title', 'FamilySize']
    result = df.copy()
    df_temp = df[features].copy()
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_temp[feature])
        result[feature] = le.transform(result[feature])
    return result 
##############################
# loading data
all_df = pd.read_csv('pre_processed_data/all_df.csv')

corrmat = all_df.corr()
f, ax = plt.subplots(figsize = (12, 9))
sns.heatmap(corrmat, vmax = 0.8, square = True)

all_df.drop(['SibSp', 'Parch'], axis = 1, inplace = True)

all_df = data_encode(all_df, ['Sex', 'Embarked','Title'])

# new feature, polynomials
def add_poly(df, features):
    for feature in features:
        col_s2 = feature + '-s2'
        col_s3 = feature + '-s3'
        col_sq = feature + '-Sq'
        df[col_s2] = df[feature] **2
        df[col_s3] = df[feature] **3
        df[col_sq] = np.sqrt(df[feature])
    return df

all_df = add_poly(all_df, ['Pclass','Age','Fare','FamilySize'])

train_df = all_df[:891]
test_df = all_df[891:]

y_train = train_df['Survived']
x_train = train_df.drop(['PassengerId','Survived'], axis = 1)
test_id = test_df['PassengerId']
x_test = test_df.drop(['PassengerId','Survived'], axis = 1)


##### 
# for nn model
def run_nn_kfold(model, x_all, y_all, n_folds = 10, print_each = False, print_head = 3):
    kf = KFold(len(y_all), n_folds)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
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

def nn_model_pred(model, x_test, y_ID):
    prediction = model.predict(x_test)
    prediction = [int(round(x[0])) for x in prediction]
    output = pd.DataFrame({'ID' : y_ID, 'Survived': prediction})
    return output    
#################################################################################

input_dim = x_train.shape[1]

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim = input_dim, activation='relu'))
    model.add(Dense(20, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

model = create_model()

# Fit the model
model.fit(x_train, y_train, epochs = 250, batch_size = 10)

# evaluate
evalu_nn_model = (model, model.evaluate(x_train, y_train), run_nn_kfold(model, x_train, y_train))
print(evalu_nn_model[1])
print(evalu_nn_model[2])

# prediction
pred_nn_df = nn_model_pred(nn_model, x_test, test_id)
pred_nn_df.head()



#################################################################################
'''
Tune Batch Size and Number of Epochs
'''

# Function to create model, required for KerasClassifier
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim = 8, activation='relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
model = KerasClassifier(build_fn=create_model, verbose=0)

# define the grid search parameters
batch_size = [10, 20, 40]
epochs = [150, 200, 250]
param_grid = dict(batch_size = batch_size, epochs = epochs)
grid = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs = 1)

grid_result = grid.fit(x_all, y_all)

# summarize result
print("Best: {} using {}".format(grid_result.best_score_, grid_result.best_params_))

#####################################################
'''
Tune the Training Optimization Algorithm
'''

# Function to create model, required for KerasClassifier
def create_model(optimizer='adam'):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer = optimizer, metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
# using the epochs and batch_size we choose base on the tuning
model = KerasClassifier(build_fn=create_model, epochs=250, batch_size=20, verbose=0)

# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(optimizer=optimizer)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)

grid_result = grid.fit(x_all, y_all)

# summarize result
print("Best: {} using {}".format(grid_result.best_score_, grid_result.best_params_))

###################################################################################
'''
Tune Learning Rate and Momentum
'''
from keras.optimizers import SGD

def create_model(learn_rate=0.01, momentum=0):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

model = KerasClassifier(build_fn=create_model, epochs=250, batch_size=20, verbose=0)

# define the grid search parameters
learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
param_grid = dict(learn_rate=learn_rate, momentum=momentum)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)

grid_result = grid.fit(x_all, y_all)

# summarize results
print("Best: {} using {}".format(grid_result.best_score_, grid_result.best_params_))

#####################################################################################
'''
Tune Network Weight Initialization
'''
# Function to create model, required for KerasClassifier
def create_model(init_mode='uniform'):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer=init_mode, activation='relu'))
    model.add(Dense(10, kernel_initializer=init_mode, activation = 'relu'))
    model.add(Dense(1, kernel_initializer=init_mode, activation='sigmoid'))
    optimizer = SGD(lr=0.001, momentum=0.9)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
model = KerasClassifier(build_fn=create_model, epochs=250, batch_size=20, verbose=0)

# define the grid search parameters
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
param_grid = dict(init_mode=init_mode)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)

grid_result = grid.fit(x_all, y_all)

# summarize results
print("Best: {} using {}".format(grid_result.best_score_, grid_result.best_params_))

####################################################################################
'''
Tune the Neuron Activation Function
'''
def create_model(activation='relu'):
    # create model
    model = Sequential()
    model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation=activation))
    model.add(Dense(10, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    optimizer = SGD(lr=0.001, momentum=0.9)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
model = KerasClassifier(build_fn=create_model, epochs=250, batch_size=20, verbose=0)

# define the grid search parameters
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
param_grid = dict(activation=activation)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)

grid_result = grid.fit(x_all, y_all)

# summarize results
print("Best: {} using {}".format(grid_result.best_score_, grid_result.best_params_))

#######################################################################################
'''
Tune Dropout Regularization
'''
# Function to create model, required for KerasClassifier
def create_model(dropout_rate=0.0, weight_constraint=0):
    # create model
    model = Sequential()
    model.add(Dropout(dropout_rate))
    model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='tanh', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, kernel_initializer='uniform', activation='relu', kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    optimizer = SGD(lr=0.001, momentum=0.9)
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
model = KerasClassifier(build_fn=create_model, epochs=250, batch_size=20, verbose=0)

# define the grid search parameters
weight_constraint = [1, 2, 3, 4, 5]
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
param_grid = dict(dropout_rate=dropout_rate, weight_constraint=weight_constraint)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)

grid_result = grid.fit(x_all, y_all)

# summarize results
print("Best: {} using {}".format(grid_result.best_score_, grid_result.best_params_))

#####################################################################################
'''
Tune the Number of Neurons in the Hidden Layer
'''

# Function to create model, required for KerasClassifier
def create_model(neurons=1):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=8, kernel_initializer='uniform', activation='linear', kernel_constraint=maxnorm(4)))
    model.add(Dropout(0.2))
    model.add(Dense(10, kernel_initializer='uniform', activation='relu', kernel_constraint=maxnorm(4)))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# create model
model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=10, verbose=0)

# define the grid search parameters
neurons = [1, 5, 10, 15, 20, 25, 30]
param_grid = dict(neurons=neurons)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)

grid_result = grid.fit(x_all, y_all)

# summarize results
print("Best: {} using {}".format(grid_result.best_score_, grid_result.best_params_))


########################################
''' try out'''

# create model
nn_model = Sequential()
nn_model.add(Dense(30, input_dim = 8, activation = 'relu'))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(10, activation = 'relu'))
nn_model.add(Dropout(0.2))
nn_model.add(Dense(1, activation = 'sigmoid'))
# Complie model
nn_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# Fit the model
nn_model.fit(x_all, y_all, epochs = 200, batch_size = 20)

# evaluate
nn_model = (nn_model, nn_model.evaluate(x_all, y_all), run_nn_kfold(nn_model, x_all, y_all))
print(nn_model[1])
print(nn_model[2])

# prediction
pred_nn_df = nn_model_pred(nn_model[0], x_test, test_id)
pred_nn_df.head()

# output
output = pred_nn_df.copy()
output = output.rename(columns = {'ID': 'PassengerId'} )
output.head(3)

output.to_csv('pre_processed_data/titanic_nn_pred2.csv', index = False)
