# Titanic ML

# importing data analysis 
import numpy as np
import pandas as pd

# importing ML modules
# Modelling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.linear_model import Perceptron,SGDClassifier

# Modelling Helpers
from sklearn.preprocessing import Imputer , Normalizer , scale
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer, accuracy_score
from sklearn import preprocessing

# deep learning
from keras.models import Sequential
from keras.layers import Dense

###############################################################################
# function for ML
# using kfold for valiadation

def run_kfold(model, x_all, y_all, n_folds = 10, print_each = False, print_head = 3):
    kf = KFold(len(y_all), n_folds)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        x_train, x_test = x_all.values[train_index], x_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        if print_each:
            print("Fold {} accuracy: {}".format(fold, accuracy))
    mean_outcome = np.mean(outcomes)
    #print(outcomes[:print_head])
    #print("Mean Accuracy: {}".format(mean_outcome))
    return mean_outcome

def model_build(no,x,y):
    ''' 
    no1: Logistic Regression |  no2: KNN |  no3: SVM
    no4: Decision Tree |  no5: Random Forest
    '''
    if no == 1:
        model = LogisticRegression()
    if no == 2:
        model = KNeighborsClassifier(n_neighbors = 3)
    if no == 3:
        model = SVC()
    if no == 4:
        model = DecisionTreeClassifier()
    if no == 5:
        model = RandomForestClassifier(n_estimators=100)
        
    model.fit(x,y)
    acc = round(model.score(x, y)*100, 2)
    
    valid_kfold_acc = run_kfold(model, x, y)
    return model, acc, valid_kfold_acc

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

###########################################################################################
###########################################################################################

# loading data
# dicide which data to load
#data = pd.read_csv('pre_processed_data/all_df_category.csv')    
data = pd.read_csv('pre_processed_data/all_df_mean_norm.csv')
#data = pd.read_csv('pre_processed_data/all_df_minMax.csv')

#### drop similar column
'''
drop SibSp, Parch
'''
#data = data.drop(['SibSp','Parch'], axis = 1)



train_df = data[:891]
test_df = data[891:]
test_id = test_df['PassengerId']

#######################################################
# get pre processed set A data

x_all = train_df.drop(['Survived','PassengerId'], axis = 1)
y_all = train_df['Survived']
x_test = test_df.drop(['Survived','PassengerId'], axis = 1)

#######################################################

# build setA model
model = []
for i in range(1,6):
    model.append(model_build(i, x_all, y_all))


######################################################

# build setA nn model

# create model
nn_model = Sequential()
nn_model.add(Dense(30, input_dim = 10, activation = 'relu'))
nn_model.add(Dense(20, activation = 'relu'))
nn_model.add(Dense(10, activation = 'relu'))
nn_model.add(Dense(1, activation = 'sigmoid'))
# Complie model
nn_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# Fit the model
nn_model.fit(x_all, y_all, epochs = 250, batch_size = 10)

# evaluate
evalu_nn_model = (nn_model, nn_model.evaluate(x_all, y_all), run_nn_kfold(nn_model, x_all, y_all))
print(evalu_nn_model[1])
print(evalu_nn_model[2])

# prediction
pred_nn_df = nn_model_pred(nn_model, x_test, test_id)
pred_nn_df.head()

# output
output = pred_nn_df.copy()
output = output.rename(columns = {'ID': 'PassengerId'} )
output.head(3)

output.to_csv('result_prediction/titanic_nn_pred3.csv', index = False)