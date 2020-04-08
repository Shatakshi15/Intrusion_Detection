# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 22:15:19 2020

@author: shatakshi
"""
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import random as r
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
features=['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted',
'num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
'is_host_login',
'is_guest_login',
'count',
'srv_count',
'serror_rate',
'srv_serror_rate',
'rerror_rate',
'srv_rerror_rate',
'same_srv_rate',
'diff_srv_rate',
'srv_diff_host_rate',
'dst_host_count',
'dst_host_srv_count',
'dst_host_same_srv_rate',
'dst_host_diff_srv_rate',
'dst_host_same_src_port_rate',
'dst_host_srv_diff_host_rate',
'dst_host_serror_rate',
'dst_host_srv_serror_rate',
'dst_host_rerror_rate',
'dst_host_srv_rerror_rate',
'class']
data = pd.read_csv('Train_data.csv', names = features, header=None)
print('The no of data points are:',data.shape[0])
print('='*40)
print('The no of features are:',data.shape[1])
print('='*40)
print('Some of the features are:',data[:10])
print('='*40)
print(data.head())
print('='*40)
output = data['class'].values
labels = set(output)
print('The different type of output labels are:',labels)
print('='*125)
print('No. of different output labels are:', len(labels))

'''for checking null value'''
print('Null values in dataset are',len(data[data.isnull().any(1)]))
data.dropna(how='any',axis=0)

'''for checking duplicate value'''
duplicateRow=data[data.duplicated()]
data.drop_duplicates(subset=features, keep='first', inplace = True)
print(data.shape)
print('='*40)

'''pickling the file'''
data.to_pickle('data.pkl')
data=pd.read_pickle('data.pkl')

'''Exploratory data analaysis'''
plt.figure(figsize=(20,15))
class_distribution = data['class'].value_counts()
class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()
sorted_yi = np.argsort(-class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1,':', class_distribution.values[i], '(', np.round((class_distribution.values[i]/data.shape[0]*100), 3), '%)')
print('='*40)

'''This function creates pairplot taking 4 features from our dataset as default parameters along with the output variable '''
def pairplot(data, label, features=[]):
    plt.figure(figsize=(20,15))
    sns.pairplot(data, hue=label, height=4, diag_kind='hist', vars=features, plot_kws={'alpha':0.6, 's':80, 'edgecolor':'k'})
    
pairplot(data, 'class', features=['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment'])

'''Train test splitting of data'''
from sklearn.model_selection import train_test_split
(X_train, X_test, Y_train, Y_test) = train_test_split(data.drop('class', axis=1), data['class'], test_size=0.25)
print('Train data')
print('X Training data shape = ',X_train.shape)
print('Y Training data shape = ',Y_train.shape)
print('='*20)
print('Test data')
print('X Testing data shape = ',X_test.shape)
print('Y Testing data shape = ',Y_test.shape)
print('='*40)

'''vectorizing categorical data using one hot encoder 
There are three categorical data:- protocol,service and flag'''

'''for protocol'''
protocol = list(X_train['protocol_type'].values)
protocol = list(set(protocol))
print('Protocol types are:', protocol)
print('='*40)

from sklearn.feature_extraction.text import CountVectorizer
one_hot = CountVectorizer(vocabulary=protocol, binary=True)
train_protocol = one_hot.fit_transform(X_train['protocol_type'].values)
test_protocol = one_hot.transform(X_test['protocol_type'].values)
print('Train protocol value after vectorizing = ',train_protocol[1].toarray())
print('='*40)
print('Train protocol shape afer vectorizing = ',train_protocol.shape)
print('='*40)

'''for service'''
service = list(X_train['service'].values)
service = list(set(service))
print('Service types are:', service)
print('='*40)
one_hot = CountVectorizer(vocabulary=service, binary=True)
train_service = one_hot.fit_transform(X_train['service'].values)
test_service = one_hot.transform(X_test['service'].values)
print('Train service value after vectorizing = ',train_service[1].toarray())
print('='*40)
print('Train service shape after vectorizing = ',train_service.shape)
print('='*40)

'''for flag'''
flag = list(X_train['flag'].values)
flag = list(set(flag))
print('flag types are:', flag)
print('='*40)
one_hot = CountVectorizer(vocabulary=flag, binary=True)
train_flag = one_hot.fit_transform(X_train['flag'].values)
test_flag = one_hot.transform(X_test['flag'].values)
print('Train flag value after vectorizing = ',test_flag[3000].toarray())
print('='*40)
print('Train flag shape after vectorizing = ',train_flag.shape)
print('='*40)

'''dropping categorical feature from train and test dataset'''
X_train.drop(['protocol_type','service','flag'], axis=1, inplace=True)
X_test.drop(['protocol_type','service','flag'], axis=1, inplace=True)

'''Applying standarisation on continous data'''
'''one by one all the continuous data are standarised here onwards'''

def feature_scaling(X_train, X_test, feature_name):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler1 = scaler.fit_transform(X_train[feature_name].values.reshape(-1,1))
    scaler2 = scaler.transform(X_test[feature_name].values.reshape(-1,1))
    return scaler1, scaler2
#duration
print('Standarised value of all the continuous data are displayed here onwards')
print('='*40)
duration1, duration2 = feature_scaling(X_train, X_test, 'duration')
print('duration',duration1[1])
print('='*40)
#src_bytes
src_bytes1, src_bytes2 = feature_scaling(X_train, X_test, 'src_bytes')
print('src_bytes1 = ',src_bytes1[1])
print('='*40)
#dst_bytes
dst_bytes1, dst_bytes2 = feature_scaling(X_train, X_test, 'dst_bytes')
print('dst_bytes1 = ',dst_bytes1[1])
print('='*40)
#wrong_fragment
wrong_fragment1, wrong_fragment2 = feature_scaling(X_train, X_test, 'wrong_fragment')
print('wrong_fragment1 = ',wrong_fragment1[1])
print('='*40)
#urgent
urgent1, urgent2 = feature_scaling(X_train, X_test, 'urgent')
print('urgent1 = ',urgent1[1])
print('='*40)
#hot
hot1, hot2 = feature_scaling(X_train, X_test, 'hot')
print('hot1 = ',hot1[1])
print('='*40)
#num_failed_logins
num_failed_logins1, num_failed_logins2 = feature_scaling(X_train, X_test, 'num_failed_logins')
print('num_failed_logins1 = ',num_failed_logins1[1])
print('='*40)
#num_compromised 
num_compromised1, num_compromised2 = feature_scaling(X_train, X_test, 'num_compromised')
print('num_compromised1 = ',num_compromised1[1])
print('='*40)
#root_shell 
root_shell1, root_shell2 = feature_scaling(X_train, X_test, 'root_shell')
print('root_shell1 = ',root_shell1[1])
print('='*40)
#su_attempted 
su_attempted1, su_attempted2 = feature_scaling(X_train, X_test, 'su_attempted')
print('su_attempted1 = ',su_attempted1[1])
print('='*40)
#num_root
num_root1, num_root2 = feature_scaling(X_train, X_test, 'num_root')
print('num_root1 = ',num_root1[1])
print('='*40)
#num_file_creations
num_file_creations1, num_file_creations2 = feature_scaling(X_train, X_test, 'num_file_creations')
print('num_file_creations1 = ',num_file_creations1[1])
print('='*40)
#num_shells
num_shells1, num_shells2 = feature_scaling(X_train, X_test, 'num_shells')
print('num_shells1 = ',num_shells1[1])
print('='*40)
#num_access_files
num_access_files1, num_access_files2 = feature_scaling(X_train, X_test, 'num_access_files')
print('num_access_files1 = ',num_access_files1[1])
print('='*40)
#num_outbound_cmds
data['num_outbound_cmds'].value_counts()
'''- We will not use 'num_outbound_cmds' feature as it has all zero values.'''
#srv_count:-
srv_count1, srv_count2 = feature_scaling(X_train, X_test, 'srv_count')
print('srv_count1 = ',srv_count1[1])
print('='*40)
#serror_rate
serror_rate1, serror_rate2 = feature_scaling(X_train, X_test, 'serror_rate')
print('serror_rate1 = ',serror_rate1[1])
print('='*40)
#srv_serror_rate
srv_serror_rate1, srv_serror_rate2 = feature_scaling(X_train, X_test, 'srv_serror_rate')
print('srv_serror_rate1 = ',srv_serror_rate1[1])
print('='*40)
#rerror_rate
rerror_rate1, rerror_rate2 = feature_scaling(X_train, X_test, 'rerror_rate')
print('rerror_rate1 = ',rerror_rate1[1])
print('='*40)
#srv_rerror_rate
srv_rerror_rate1, srv_rerror_rate2 = feature_scaling(X_train, X_test, 'srv_rerror_rate')
print('srv_rerror_rate1 = ',srv_rerror_rate1[1])
print('='*40)
#same_srv_rate
same_srv_rate1, same_srv_rate2 = feature_scaling(X_train, X_test, 'same_srv_rate')
print('same_srv_rate1 = ',same_srv_rate1[1])
print('='*40)
#diff_srv_rate
diff_srv_rate1, diff_srv_rate2 = feature_scaling(X_train, X_test, 'diff_srv_rate')
print('diff_srv_rate1 = ',diff_srv_rate1[1])
print('='*40)
#srv_diff_host_rate
srv_diff_host_rate1, srv_diff_host_rate2 = feature_scaling(X_train, X_test, 'srv_diff_host_rate')
print('srv_diff_host_rate1 = ',srv_diff_host_rate1[1])
print('='*40)
#dst_host_count:-
dst_host_count1, dst_host_count2 = feature_scaling(X_train, X_test, 'dst_host_count')
print('dst_host_count1 = ',dst_host_count1[1])
print('='*40)
#dst_host_srv_count:-
dst_host_srv_count1, dst_host_srv_count2 = feature_scaling(X_train, X_test, 'dst_host_srv_count')
print('dst_host_srv_count1 = ',dst_host_srv_count1[1])
print('='*40)
#dst_host_same_srv_rate:-
dst_host_same_srv_rate1, dst_host_same_srv_rate2= feature_scaling(X_train, X_test,'dst_host_same_srv_rate')
print('dst_host_same_srv_rate1 = ',dst_host_same_srv_rate1[1])
print('='*40)
#dst_host_diff_srv_rate:-
dst_host_diff_srv_rate1, dst_host_diff_srv_rate2 = feature_scaling(X_train, X_test,'dst_host_diff_srv_rate')
print('dst_host_diff_srv_rate1 = ',dst_host_diff_srv_rate1[1])
print('='*40)
#dst_host_same_src_port_rate:-
dst_host_same_src_port_rate1, dst_host_same_src_port_rate2 = feature_scaling(X_train, X_test,'dst_host_same_src_port_rate')
print('dst_host_same_src_port_rate1 = ',dst_host_same_src_port_rate1[1])
print('='*40)
#dst_host_srv_diff_host_rate:-
dst_host_srv_diff_host_rate1, dst_host_srv_diff_host_rate2 = feature_scaling(X_train, X_test,'dst_host_srv_diff_host_rate')
print('dst_host_srv_diff_host_rate1 = ',dst_host_srv_diff_host_rate1[1])
print('='*40)
#dst_host_serror_rate:-
dst_host_serror_rate1, dst_host_serror_rate2 = feature_scaling(X_train, X_test, 'dst_host_serror_rate')
print('dst_host_serror_rate1 = ',dst_host_serror_rate1[1])
print('='*40)
#dst_host_srv_serror_rate:-
dst_host_srv_serror_rate1, dst_host_srv_serror_rate2 = feature_scaling(X_train, X_test,'dst_host_srv_serror_rate')
print('dst_host_srv_serror_rate1 = ',dst_host_srv_serror_rate1[1])
print('='*40)
#dst_host_rerror_rate:-
dst_host_rerror_rate1, dst_host_rerror_rate2 = feature_scaling(X_train, X_test, 'dst_host_rerror_rate')
print('dst_host_rerror_rate1 = ',dst_host_rerror_rate1[1])
print('='*40)
#dst_host_srv_rerror_rate:-
dst_host_srv_rerror_rate1, dst_host_srv_rerror_rate2 = feature_scaling(X_train, X_test,'dst_host_srv_rerror_rate')
print('dst_host_srv_rerror_rate1 = ',dst_host_srv_rerror_rate1[1])
print('='*40)
#num_failed_logins :-
num_failed_logins1, num_failed_logins2 = feature_scaling(X_train, X_test, 'num_failed_logins')
print('num_failed_logins1 = ',num_failed_logins1[1])
print('='*40)
#land:-
land1, land2 = np.array([X_train['land'].values]), np.array([X_test['land'].values])
print('land1 shape = ',land1.shape)
print('='*40)
#is_host_login :-
is_host_login1, is_host_login2 = np.array([X_train['is_host_login'].values]), np.array([X_test['is_host_login'].values])
print('is_host_login1 shape = ',is_host_login1.shape)
print('='*40)
#is_guest_login :-
is_guest_login1, is_guest_login2 = np.array([X_train['is_guest_login'].values]), np.array([X_test['is_guest_login'].values])
print('is_guest_login1 shape = ',is_guest_login1.shape)
print('='*40)
#logged_in :-
logged_in1, logged_in2 = np.array([X_train['logged_in'].values]), np.array([X_test['logged_in'].values])
print('logged_in1 shape = ',logged_in1.shape)
print('='*40)
#count:-
count1, count2 = feature_scaling(X_train, X_test, 'count')
print('count1 = ',count1[1])
print('='*40)
#dst_host_diff_srv_rate:-
dst_host_diff_srv_rate1, dst_host_diff_srv_rate2 = feature_scaling(X_train, X_test,'dst_host_diff_srv_rate')
print('dst_host_diff_srv_rate1 = ',dst_host_diff_srv_rate1[1])
print('='*40)

'''Merging categorical and continous data'''

from scipy.sparse import hstack
X_train_1 = hstack((duration1, train_protocol, train_service, train_flag, src_bytes1,
dst_bytes1, land1.T, wrong_fragment1, urgent1, hot1,
num_failed_logins1, logged_in1.T, num_compromised1, root_shell1,
su_attempted1, num_root1, num_file_creations1, num_shells1,
num_access_files1, is_host_login1.T,
is_guest_login1.T, count1, srv_count1, serror_rate1,
srv_serror_rate1, rerror_rate1, srv_rerror_rate1, same_srv_rate1,
diff_srv_rate1, srv_diff_host_rate1, dst_host_count1,
dst_host_srv_count1, dst_host_same_srv_rate1,
dst_host_diff_srv_rate1, dst_host_same_src_port_rate1,
dst_host_srv_diff_host_rate1, dst_host_serror_rate1,
dst_host_srv_serror_rate1, dst_host_rerror_rate1,
dst_host_srv_rerror_rate1))

print('X_train shape after merging categorical and continuous data = ',X_train_1.shape)
print('='*40)

X_test_1 = hstack((duration2, test_protocol, test_service, test_flag, src_bytes2,
dst_bytes2, land2.T, wrong_fragment2, urgent2, hot2,
num_failed_logins2, logged_in2.T, num_compromised2, root_shell2,
su_attempted2, num_root2, num_file_creations2, num_shells2,
num_access_files2, is_host_login2.T,
is_guest_login2.T, count2, srv_count2, serror_rate2,
srv_serror_rate2, rerror_rate2, srv_rerror_rate2, same_srv_rate2,
diff_srv_rate2, srv_diff_host_rate2, dst_host_count2,
dst_host_srv_count2, dst_host_same_srv_rate2,
dst_host_diff_srv_rate2, dst_host_same_src_port_rate2,
dst_host_srv_diff_host_rate2, dst_host_serror_rate2,
dst_host_srv_serror_rate2, dst_host_rerror_rate2,
dst_host_srv_rerror_rate2))

print('X_train shape after merging categorical and continuous data = ',X_test_1.shape)
print('='*40)

from sklearn.externals import joblib
joblib.dump(X_train_1,'X_train_1.pkl')
joblib.dump(X_test_1,'X_test_1.pkl')
X_train_1 = joblib.load('X_train_1.pkl')
X_test_1 = joblib.load('X_test_1.pkl')

joblib.dump(Y_train,'Y_train.pkl')
joblib.dump(Y_test,'Y_test.pkl')
Y_train = joblib.load('Y_train.pkl')
Y_test = joblib.load('Y_test.pkl')

'''few utility functions to measure performance of the model used for training'''
#function1
import datetime as dt
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
'''
This function plots the confusion matrix heatmap using the actual and predicted values.
'''
def confusion_matrix_func(Y_test, y_test_pred):

    C = confusion_matrix(Y_test, y_test_pred)
    cm_df = pd.DataFrame(C)
    labels = ['normal','anomaly']
    plt.figure(figsize=(20,15))
    sns.set(font_scale=1.4)
    sns.heatmap(cm_df, annot=True, annot_kws={"size":12}, fmt='g', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.show()

'''This function computes the performance scores on the train and test data.'''
def model(model_name, X_train, Y_train, X_test, Y_test):

    print('Fitting the model and prediction on train data:')
    start = dt.datetime.now()
    model_name.fit(X_train, Y_train)
    y_tr_pred = model_name.predict(X_train)
    print('Completed')
    print('Time taken:',dt.datetime.now()-start)
    print('='*60)
    results_tr = dict()
    y_tr_pred = model_name.predict(X_train)
    results_tr['precision'] = precision_score(Y_train, y_tr_pred, average='weighted')
    results_tr['recall'] = recall_score(Y_train, y_tr_pred, average='weighted')
    results_tr['f1_score'] = f1_score(Y_train, y_tr_pred, average='weighted')
    results_test = dict()
    print('Prediction on test data:')
    start = dt.datetime.now()
    y_test_pred = model_name.predict(X_test)
    print('Completed')
    print('Time taken:',dt.datetime.now()-start)
    print('='*60)
    print('Performance metrics:')
    print('='*60)
    print('Confusion Matrix is:')
    confusion_matrix_func(Y_test, y_test_pred)
    print('='*60)
    results_test['precision'] = precision_score(Y_test, y_test_pred, average='weighted')
    print('Precision score is:')
    print(precision_score(Y_test, y_test_pred, average='weighted'))
    print('='*60)
    results_test['recall'] = recall_score(Y_test, y_test_pred, average='weighted')
    print('Recall score is:')
    print(recall_score(Y_test, y_test_pred, average='weighted'))
    print('='*60)
    results_test['f1_score'] = f1_score(Y_test, y_test_pred, average='weighted')
    print('F1-score is:')
    print(f1_score(Y_test, y_test_pred, average='weighted'))
# add the trained model to the results
    results_test['model'] = model
    return results_tr, results_test

'''
This function prints all the grid search attributes
'''

def print_grid_search_attributes(model):

    print('---------------------------')
    print('| Best Estimator |')
    print('---------------------------')
    print('\n\t{}\n'.format(model.best_estimator_))
# parameters that gave best results while performing grid search
    print('---------------------------')
    print('| Best parameters |')
    print('---------------------------')
    print('\tParameters of best estimator : \n\n\t{}\n'.format(model.best_params_))
# number of cross validation splits
    print('----------------------------------')
    print('| No of CrossValidation sets |')
    print('----------------------------------')
    print('\n\tTotal numbre of cross validation sets: {}\n'.format(model.n_splits_))
# Average cross validated score of the best estimator, from the Grid Search
    print('---------------------------')
    print('| Best Score |')
    print('---------------------------')
    print('\n\tAverage Cross Validate scores of best estimator :\n\n\t{}\n'.format(model.best_score_))
 
'''This function computes the TPR and FPR scores using the actual and predicetd values'''

def tpr_fpr_func(Y_tr, Y_pred):
    results = dict()
    Y_tr = Y_tr.to_list()
    tp = 0; fp = 0; positives = 0; negatives = 0;tpr=0;fpr=0; length = len(Y_tr)
    for i in range(length):
        if Y_tr[i]=='normal':
            positives += 1
        else:
                negatives += 1
    for i in range(len(Y_pred)):
        if Y_tr[i]=='normal' and Y_pred[i]=='normal':
            tp += 1
        elif Y_tr[i]!='normal' and Y_pred[i]=='normal':
            fp += 1
    if(positives!=0):
        tpr = tp/positives
    if(negatives!=0):
        fpr = fp/negatives
    results['tp'] = tp; results['tpr'] = tpr; results['fp'] = fp; results['fpr'] = fpr
    return results

'''decision tree algorithm for training and testing the data'''

hyperparameter = {'max_depth':[5, 10, 20, 50, 100, 500], 'min_samples_split':[5, 10, 100, 500]}
decision_tree = DecisionTreeClassifier(criterion='gini', splitter='best',class_weight='balanced')
decision_tree_grid = GridSearchCV(decision_tree, param_grid=hyperparameter, cv=3, verbose=1, n_jobs=-1)
decision_tree_grid_results_train, decision_tree_grid_results_test = model(decision_tree_grid,X_train_1.toarray(), Y_train, X_test_1.toarray(), Y_test)
print_grid_search_attributes(decision_tree_grid)
joblib.dump(decision_tree_grid.best_estimator_, 'decision_tree_gs.pkl')
dt_gs = decision_tree_grid.best_estimator_

'''predicting on train data'''
y_train_pred = dt_gs.predict(X_train_1.toarray())

'''predicting on test data'''
y_test_pred = dt_gs.predict(X_test_1.toarray())

dt_tpr_fpr_train = tpr_fpr_func(Y_train, y_train_pred)
dt_tpr_fpr_test = tpr_fpr_func(Y_test, y_test_pred)

print('''Decision tree grid result for training values''')
print(decision_tree_grid_results_train)
print('='*40)

print('''display of tpr and fpr values for training data ''')
print(dt_tpr_fpr_train)
print('='*40)

print('''Decision tree grid result for testing values''')
print(decision_tree_grid_results_test)
print('='*40)

print('''display of tpr and fpr values for testing data ''')
print(dt_tpr_fpr_test)
print('='*40)




