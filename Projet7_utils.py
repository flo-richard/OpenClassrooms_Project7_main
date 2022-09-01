import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
import imblearn

from sklearn.impute import SimpleImputer


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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    