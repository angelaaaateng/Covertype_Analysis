'''Import Libraries'''
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from pprint import pprint
from sklearn.externals.six import StringIO
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import os

# os.chdir('/Users/angelateng/Documents/GitHub/Projects/Covertype_Prediction/Archive/Data')
# print(os.getcwd())
# print(__name__)
# print('Directory navigated')
# input = open("./covtype.data")


#doesn't work unless if name == main is the first module
#from app.py import transform_view

def read_data(csv_file):
    '''
    Read Data
    '''
    data = csv_file
    # data = pd.DataFrame([csv_file], index=None)
    # data = pd.read_data(csv_file, header=None)
    # set column names
    cols = ['elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology',
       'vertical_distance_to_hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points',
       'Wilderness_Area_1', 'Wilderness_Area_2', 'Wilderness_Area_3', 'Wilderness_Area_4',
       'Soil_Type_1',
        'Soil_Type_2',
        'Soil_Type_3',
        'Soil_Type_4',
        'Soil_Type_5',
        'Soil_Type_6',
        'Soil_Type_7',
        'Soil_Type_8',
        'Soil_Type_9',
        'Soil_Type_10',
        'Soil_Type_11',
        'Soil_Type_12',
        'Soil_Type_13',
        'Soil_Type_14',
        'Soil_Type_15',
        'Soil_Type_16',
        'Soil_Type_17',
        'Soil_Type_18',
        'Soil_Type_19',
        'Soil_Type_20',
        'Soil_Type_21',
        'Soil_Type_22',
        'Soil_Type_23',
        'Soil_Type_24',
        'Soil_Type_25',
        'Soil_Type_26',
        'Soil_Type_27',
        'Soil_Type_28',
        'Soil_Type_29',
        'Soil_Type_30',
        'Soil_Type_31',
        'Soil_Type_32',
        'Soil_Type_33',
        'Soil_Type_34',
        'Soil_Type_35',
        'Soil_Type_36',
        'Soil_Type_37',
        'Soil_Type_38',
        'Soil_Type_39',
        'Soil_Type_40',
       'Cover_Type']
    data.columns = cols
    #print(data['Cover_Type'])
    print('* Data loaded - preprocessing...')
    cov_dummy = pd.get_dummies(data['Cover_Type'])
    df4 = pd.concat([cov_dummy, data], axis = 1)
    df4_column_names = list(df4.columns)
    df4_column_names.remove('Cover_Type')
    print('* Data loaded - preprocessing complete')
    #pprint(data.columns)
    #print(df4)
    #print(read_data)
    # print(df4_column_names)
    return(data, df4, df4_column_names);

#data, df4, df4_column_names = read_data("./covtype.data")


def normalize_data(df4, df4_column_names):
    x = df4.loc[:, df4.columns != 'Cover_Type'].values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(data=x_scaled, columns=df4_column_names)
    #print(df_normalized)
    df_normalized_w_target = pd.concat([df_normalized, df4['Cover_Type']], axis=1)
    df_dummy = df_normalized_w_target
    df_dummy = df_dummy.drop(['Cover_Type'], axis=1)
    X_test_new=df_normalized_w_target[list(df_normalized_w_target.columns)[7:-1]]
    y_test_new=df_normalized_w_target[list(df_normalized_w_target.columns)[-1]]
    print('* Data Normalized - preprocessing')
    # print(y_test)
    return(df_normalized, df_normalized_w_target, X_test_new, y_test_new)

#normalize_data(df4, df4_column_names)

def preprocess(csv_file):
    data, df4, df4_column_names = read_data(csv_file)
    df_normalized, df_normalized_w_target, X_test_new, y_test_new = normalize_data(df4, df4_column_names)
    print('* Data Preprocessing Complete')
    return(data, df4, df4_column_names, df_normalized, df_normalized_w_target, X_test_new, y_test_new)

#preprocess("./covtype.data")

if __name__ == "__main__":
    '''
    Navigate Directory
    '''
    print('* Data Preprocessing Running')
    #print("* Navigating through directory")
