import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import imblearn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def diff_lists(L1, L2):
    """Returns the elements that are in a list L1 but not in another list L2, and vice versa"""
    
    diff_12 = list(set(L1) - set(L2))  #in L1 but not in L2
    diff_21 = list(set(L2) - set(L1))  #in L2 but not in L1
    
    return diff_12, diff_21




def cat_to_binary(series):
    
    L_categ = series.unique()
        
    for count, value in enumerate(L_categ):        
        series = series.replace(value, count)
            
    return series


def imput(df, L_features, func):
    
    imputer = SimpleImputer(strategy=func)
    df[L_features] = imputer.fit_transform(df[L_features])
    
    return pd.DataFrame(df[L_features], columns=L_features)



def preprocess_data(X, y, list_cat):
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    
    names_cont = X.drop(list_cat, axis=1).columns.tolist()
    
    indices_train = X_train.index
    indices_test = X_test.index
    
    X_cat_names = X[list_cat]
    
    
    X_cat_train = X_train[list_cat].values
    X_cat_test = X_test[list_cat].values

    X_cont_train = X_train.drop(list_cat, axis=1).values
    X_cont_test = X_test.drop(list_cat, axis=1).values
    
        # categ variables
        
    names_cat = []
    
    imputer_cat = SimpleImputer(strategy='most_frequent')
    imputer_cat.fit(X_cat_train)
    
    X_cat_train = pd.DataFrame(imputer_cat.transform(X_cat_train), columns=X_cat_names.columns.tolist())
    X_cat_test = pd.DataFrame(imputer_cat.transform(X_cat_test), columns=X_cat_names.columns.tolist())
    
    X_cat_names = pd.DataFrame(imputer_cat.transform(X_cat_names), columns=X_cat_names.columns.tolist())
    
    for i in list_cat:
        #print("i :", i)
        for j in X_cat_names[i].unique():
            #print("j :", j)
            names_cat.append(i + ':' + j)
    
    steps_cat = [
        ('OHE', OneHotEncoder()),
        ('Scaling', StandardScaler(with_mean=False))
    ]

    pipe_cat = Pipeline(steps=steps_cat)
    pipe_cat.fit(X_cat_train, y_train)

    X_cat_train = pd.DataFrame(pipe_cat.transform(X_cat_train).todense(), columns=names_cat)
    X_cat_test = pd.DataFrame(pipe_cat.transform(X_cat_test).todense(), columns=names_cat)
    
    
    
        # continuous variables
    
    names_cont = X.drop(list_cat, axis=1).columns.tolist()

    
    steps_cont = [
        ('imputer', SimpleImputer(strategy='mean')),
        ('Scaling', StandardScaler())
    ]

    pipe_cont = Pipeline(steps=steps_cont)
    pipe_cont.fit(X_cont_train, y_train)

    X_cont_train = pd.DataFrame(pipe_cont.transform(X_cont_train), columns=names_cont)
    X_cont_test = pd.DataFrame(pipe_cont.transform(X_cont_test), columns=names_cont)
    
    
    
        # aggregate cat and cont
    
    X_train = pd.concat([X_cont_train, X_cat_train], axis=1)
    X_train = X_train.set_index(indices_train)
    
    X_test = pd.concat([X_cont_test, X_cat_test], axis=1)
    X_test = X_test.set_index(indices_test)
    
    
    return X_train, X_test, y_train, y_test
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    