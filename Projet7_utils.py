import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import imblearn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, fbeta_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipe

import lime
from lime import lime_tabular
import random




def diff_lists(L1, L2):
    """Returns the elements that are in a list L1 but not in another list L2, and vice versa"""
    
    diff_12 = list(set(L1) - set(L2))  #in L1 but not in L2
    diff_21 = list(set(L2) - set(L1))  #in L2 but not in L1
    
    return diff_12, diff_21


def set_0_to_1(series):
    """Swaps 0s and 1s in a series"""
    series = series.replace({
        0: 1,
        1: 0
    })

    return series


def set_outlier_nan(series):
    """identifies outliers in a series via interquartile analysis and sets the values to NaN """
    

    try:
        series.sort_values(ascending=True)
        Q1 = series.quantile(0.25) #First quartile
        Q3 = series.quantile(0.75) #Third quartile

        #print('Q1 = ', Q1)    
        #print('Q3 = ', Q3)

        iqr = Q3 - Q1  #Interquartile range
        lower = Q1 - 1.5*iqr #lower bound
        upper = Q3 + 1.5*iqr  #upper bound
        
        for i in range(len(series)):    #loop over the elements in the series
            if series.iloc[i] > upper or series.iloc[i] < lower:  #if the element is out of bounds                
                series.iloc[i]=float('nan')  #set to NaN
                
        return series
    
    except TypeError:
        return series




def imput(df, L_features, func):
    """Trains an imputer on a list of features. Imputes the values with a given method (mean, median, most frequent, etc)"""
    
    imputer = SimpleImputer(strategy=func)
    imputer.fit(df[L_features])
    df[L_features] = imputer.transform(df[L_features])
    
    return pd.DataFrame(df[L_features], columns=L_features), imputer


def label_enc(df, list_feat_idx):
    """Trains label encoders for categorical variables. Saves them in a dictionary and returns it, as well as a map of the labels"""

    classes_names_dict = {} # label map
    transformers_dict = {} # label encoders

    for feature in list_feat_idx:
        le = LabelEncoder()
        le.fit(df.iloc[:, [feature]].values.ravel())

        classes_names_dict[feature] = le.classes_

        df.iloc[:, [feature]] = le.transform(df.iloc[:, [feature]].values.ravel())

        transformers_dict[df.columns.tolist()[feature]] = le
        #df[feature].astype('category') # set the feature as a category in the dataframe

        

    return df, classes_names_dict, transformers_dict







def train_model(data, classifier, par_grid, scorer):
    """Optimizes a classifier with a GridSearch + corss validation. Trains and returns it
    data : dictionary containing the training and testing sets (input and target)
    classifier : model to train
    par_grid : grid of parameters to test in the GridSearch
    scorer : metric to evaluate the model performances"""

    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    if classifier == LogisticRegression(): # if the model is not tree based, the data need to be scaled
        model = imbpipe(steps=[
            ('OverSampling', RandomOverSampler(random_state=42)),
            #('UnderSampling', RandomUnderSampler(random_state=42)),
            ('Scaling', StandardScaler()),
            ('classification', classifier)
        ])

    else : # if it is tree based, the data do not need to be scaled
        model = imbpipe(steps=[
            ('OverSampling', RandomOverSampler(random_state=42)),
            #('UnderSampling', RandomUnderSampler(random_state=42)),
            #('Scaling', StandardScaler()),
            ('classification', classifier)
        ])

    model_trained = GridSearchCV(
        model,
        param_grid=par_grid,
        scoring=scorer,
        verbose=1,
        n_jobs=4
    )

    model_trained.fit(X_train, y_train)

    print('Best hyperparams: ', model_trained.best_params_)
    print('Best mean score: ', model_trained.best_score_)
    print('std: ', model_trained.cv_results_['std_test_score'][model_trained.best_index_])

    return model_trained




def score_func(y_true, y_pred, beta):
    """Custom metric function. Uses the f-beta score"""

    y = pd.concat([pd.DataFrame(y_true, columns=['TARGET']).reset_index(), pd.DataFrame(y_pred, columns=['Prediction'])], axis=1)
    #display(y)

    A = y[(y.TARGET==1) & (y.Prediction==1)].count()[0] # correctly predicted as able to pay
    B = y[(y.TARGET==0) & (y.Prediction==1)].count()[0] # predicted as able to pay, but unable in reality -> major error, big coeff
    C = y[(y.TARGET==1) & (y.Prediction==0)].count()[0] # predicted as unable to pay, but able in reality -> minor error, small coeff
    D = y[(y.TARGET==0) & (y.Prediction==0)].count()[0] # correctly predicted as unable to pay

    print('Total count: ', len(y))
    print("Correctly predicted as able to pay :", A, "/", y.loc[y['TARGET']==1]['TARGET'].count())
    print("Predicted as unable to pay, but able in reality :", C)
    print("Correctly predicted as unable to pay :", D, "/", y.loc[y['TARGET']==0]['TARGET'].count())
    print("Predicted as able to pay, but unable in reality :", B)    
    
    
    score = fbeta_score(y_true, y_pred, beta=beta)
    print("beta: ", beta)
    print('fbeta score: ', score)
    return score





def local_feat_imp(idx, X_train, X_test, y_test, categorical_features_idxs, categorical_names_dict, model):
    """Computes the local feature importances with LIME of a given individual in the data set
    X_train : training dataset
    X_test, y_test : test dataset
    idx : index of the individual to be evaluated. Picked in the test dataset
    categorical_features_idxs : list of the indices of the categorical features in the dataset
    categorical_names_dict : map of the categorical features labels
    """
    
    explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,
        mode='classification',
        feature_names=X_train.columns.to_list(),
        categorical_features=categorical_features_idxs,
        categorical_names=categorical_names_dict,
        class_names=[0, 1]
    )

    pred = model.predict(X_test.loc[idx].values.reshape(1, -1))

    print("Prediction : ", pred)
    print("Actual :     ", y_test[idx])

    exp = explainer.explain_instance(
        X_test.loc[idx],
        model.predict_proba,
        num_features=6
    )

    exp.show_in_notebook()

    return explainer, exp