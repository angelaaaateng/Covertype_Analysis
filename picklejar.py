import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from imblearn.datasets import make_imbalance
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


def hyper_param_rf_pickle(X_test, y_test, model):
    # rfc = RandomForestClassifier(n_estimators=300, max_depth=20,
    #  min_samples_split=2, min_samples_leaf=1, bootstrap=True, random_state=42)
    # rf_model2 = load('./grid_search_optimal.joblib')
    rfc = model
    print('* Joblib model loaded -- picklejar')

    y_pred = rfc.predict(X_test)
    # print ("Random Forest Train Accuracy Baseline After Grid Search:", metrics.accuracy_score(y_train, grid_search_optimal.predict(X_train)))
    print ("Random Forest Test Accuracy Baseline After Grid Search:", metrics.accuracy_score(y_test, rfc.predict(X_test)))
    conf_mat = confusion_matrix(y_test,y_pred)
    class_rept = classification_report(y_test,y_pred )
    print(confusion_matrix(y_test,y_pred ))
    print(classification_report(y_test,y_pred ))
    # rfc_train_acc = metrics.accuracy_score(y_train, rfc.predict(X_train))
    rfc_test_acc = metrics.accuracy_score(y_test, rfc.predict(X_test))
    print("* Dumping RF_model")
    rf_model = dump(rfc, './grid_search_optimal.joblib')
    print("* RF model dumped!")
    # saved_model = pickle.dumps(grid_search_optimal)
    print("* Saving results in an image in picklejar...")
    mpl.style.use('seaborn')
    print("* Style set as seaborn")

    df_mat = pd.DataFrame(conf_mat)
    print("* Saved conf mat in df")
    print(df_mat)


    fig = plt.figure()
    print("* fig plotted in picklejar")
    plt.clf()
    #close figure
    print("* Plot rfc")
    cmap = sns.cubehelix_palette(light=1, as_cmap=True)
    print("* Set SNS hue")
    res = sns.heatmap(df_mat, annot=True, vmin=0.0, vmax=100.0, fmt='.2f', cmap=cmap)
    # res.invert_yaxis()

    # plt.matshow(conf_mat)
    print("* confmat plotted in picklejar")
    plt.title('Confusion Matrix')
    print("* title plotted in picklejar")
    # plt.show()
    print("* Plot printed -- now saving")
    plt.savefig('./Static/confusion_matrix.png')
    print("* Figure saved!")
    plt.close()
    print("* plt closed")

    print("* File pickled using joblib -- picklejar process complete")
    return(rfc_test_acc, y_pred, class_rept, conf_mat)

if __name__ == "__main__":
    '''
    Navigate Directory
    '''
    #print("* Navigating through directory")
    #os.chdir('/Users/angelateng/Documents/GitHub/Projects/Covertype_Prediction/Data')
    #print(os.getcwd())
    #print(__name__)
    print('Directory navigated')
    #input = open("./covtype.data")
