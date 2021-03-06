

'''Import Libraries'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier


from sklearn import metrics

from sklearn.metrics import classification_report,confusion_matrix
import sklearn.model_selection as model_selection
from IPython.display import Image
from sklearn.externals.six import StringIO




from sklearn.model_selection import validation_curve
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


from imblearn.metrics import classification_report_imbalanced
from sklearn import tree


import os

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns



def hyper_param_rf_predict(X_train, y_train, X_test, y_test):
    rfc = RandomForestClassifier()
    #rfc.fit(X_train, y_train)
    n_optimal_param_grid = {
    'bootstrap': [True],
    'max_depth': [20], #setting this so as not to create a tree that's too big
    #'max_features': [2, 3, 4, 10],
    'min_samples_leaf': [1],
    'min_samples_split': [2],
    'n_estimators': [300]
    }
    grid_search_optimal = GridSearchCV(estimator = rfc, param_grid = n_optimal_param_grid,
                          cv = 3, n_jobs = -1, verbose = 2)
    #grid_search_optimal.fit(X_train, y_train)
    rfc_pred_gs = grid_search_optimal.predict(X_test)
    print ("Random Forest Train Accuracy Baseline After Grid Search:", metrics.accuracy_score(y_train, grid_search_optimal.predict(X_train)))
    print ("Random Forest Test Accuracy Baseline After Grid Search:", metrics.accuracy_score(y_test, grid_search_optimal.predict(X_test)))
    print(confusion_matrix(y_test,rfc_pred_gs))
    print(classification_report(y_test,rfc_pred_gs))
    rfc_train_acc = metrics.accuracy_score(y_train, rfc.predict(X_train))
    rfc_test_acc = metrics.accuracy_score(y_test, rfc.predict(X_test))


    conf_mat = confusion_matrix(y_test_new,y_pred)
    class_rept = classification_report(y_test_new,y_pred )
    fig = plt.figure()
    plt.matshow(conf_mat)
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig('confusion_matrix.jpg')

    # plt.
    return(rfc_train_acc, rfc_test_accs)
