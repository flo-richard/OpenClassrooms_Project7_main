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


def set_outlier_nan(series):

    #print(series)

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
    
    imputer = SimpleImputer(strategy=func)
    imputer.fit(df[L_features])
    df[L_features] = imputer.transform(df[L_features])
    
    return pd.DataFrame(df[L_features], columns=L_features), imputer


def label_enc(df, list_feat_idx):

    classes_names_dict = {} 
    transformers_dict = {}

    for feature in list_feat_idx:
        le = LabelEncoder()
        le.fit(df.iloc[:, [feature]].values.ravel())

        classes_names_dict[feature] = le.classes_

        df.iloc[:, [feature]] = le.transform(df.iloc[:, [feature]].values.ravel())
        #le_name = 'LE_' + df.columns.tolist()[feature]

        transformers_dict[df.columns.tolist()[feature]] = le
        df.iloc[:, [feature]] = df.iloc[:, [feature]].astype('category')

        

    return df, classes_names_dict, transformers_dict





def set_0_to_1(series):

    series = series.replace({
        0: 1,
        1: 0
    })

    return series



def train_model(data, classifier, par_grid, scorer):


    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']

    model = imbpipe(steps=[
        ('OverSampling', RandomOverSampler(random_state=42)),
        #('UnderSampling', RandomUnderSampler(random_state=42)),
        ('Scaling', StandardScaler()),
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

    return exp