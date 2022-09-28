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
    df[L_features] = imputer.fit_transform(df[L_features])
    
    return pd.DataFrame(df[L_features], columns=L_features)


def categorize(df, list_cat_bin, list_dict_bin, list_cat_ordinal, list_dict_ordinal, list_cat_nominal):

    #BINARY FEATURES
    for i in range(len(list_cat_bin)):
        
        #df[list_cat_bin[i]] = pd.Categorical(df[col], ordered=False)
        #df[col] = LabelEncoder().fit_transform(df[col])
        #df[col] = df[col].astype('category')

        df[list_cat_bin[i]] = df[list_cat_bin[i]].replace(list_dict_bin[i])
        df[list_cat_bin[i]] = df[list_cat_bin[i]].astype('category')

    #ORDINAL FEATURES
    for i in range(len(list_cat_ordinal)):
        df[list_cat_ordinal[i]] = df[list_cat_ordinal[i]].replace(list_dict_ordinal[i])
        df[list_cat_ordinal[i]] = df[list_cat_ordinal[i]].astype('category')

    #NOMINAL FEATURES -> OHE

    for col in list_cat_nominal:

        ohe = OneHotEncoder(sparse=False)
        df_nominal_temp = pd.DataFrame(ohe.fit_transform(df[col].values.reshape(-1, 1)))

        #display(df_nominal_temp)
        #print(ohe.get_feature_names_out([col]))
        df_nominal_temp.columns = ohe.get_feature_names_out([col])
        df_nominal_temp = df_nominal_temp.set_index(df.index)
        df = pd.concat([df, df_nominal_temp], axis=1)
    
    df = df.drop(list_cat_nominal, axis=1)
    return df


def set_0_to_1(series):

    series = series.replace({
        0: 1,
        1: 0
    })

    return series


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



def train_model(model, par_grid, X_train, y_train, scorer):

    model_train = GridSearchCV(
        model,
        param_grid=par_grid,
        scoring=scorer,
        n_jobs=4,
        verbose=2,
        error_score='raise'
    )

    model_train.fit(X_train.values, y_train.values)

    print('Best hyperparams: ', model_train.best_params_)
    print('Best mean score: ', model_train.best_score_)
    print('std: ', model_train.cv_results_['std_test_score'][model_train.best_index_])

    return model_train

    #model_predict = model_train.predict(X_test.values)

    #return model_predict



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
    
    
    #weighted_error = (.8*B + .2*C) / (A + B + C + D)
    #weighted_error = .8*B/(A+C) + .2*C/(B+D)

    score = fbeta_score(y_true, y_pred, beta=beta)
    print("beta: ", beta)
    print('fbeta score: ', score)
    return score

    
    
    
    
    
    
    
def train_try(X, y, classifier, par_grid, scorer):


    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

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

    return model_trained, X_test, y_test


    
    
    
    
    
    
    
    
    
    
    