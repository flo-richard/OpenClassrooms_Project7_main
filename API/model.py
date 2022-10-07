import os
import pandas as pd
from pydantic import BaseModel
import dill as pickle
import lime
import numpy as np

#PATH = "E:/OpenClassrooms/Projet7/Data"

#class Id(BaseModel):
#    id: int

class ScoringModel:
    def __init__(self):

        

        self.list_features = pickle.load(open('Pickled_objects/features.pkl', 'rb'))
        self.all_features = self.list_features['all_features']
        self.continuous_features = self.list_features['continuous_features']
        self.categorical_features = self.list_features['categorical_features']        

        self.model = pickle.load(open('Pickled_objects/xgb.pkl', 'rb')) # load trained model

        self.transformers = pickle.load(open('Pickled_objects/transformers.pkl', 'rb')) #load transformers
        self.imputers = pickle.load(open('Pickled_objects/imputers.pkl', 'rb')) #load imputers
        self.list_transformers = [i for i in self.transformers]  #list of transformers names (columns to be transformed)

        self.explainer = pickle.load(open('Pickled_objects/explainer.pkl', 'rb')) #load lime explainer



    def preprocessing(self, df: pd.DataFrame, req_info: dict):

        # vérifier si les clés dans self.features sont bien dans payload
        # formater payload au bon format (bonnes clés, avec bonnes valeurs)

        # Feature engineering

        df['INCOME_CREDIT_RATE'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
        df['ANNUITY_INCOME_RATE'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
        df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']

        #print('YOLOOO ', df['INCOME_CREDIT_RATE'].values[0])

        req_info['INCOME_CREDIT_RATE'] = df['INCOME_CREDIT_RATE'].values[0]
        req_info['ANNUITY_INCOME_RATE'] = df['ANNUITY_INCOME_RATE'].values[0]
        req_info['PAYMENT_RATE'] = df['PAYMENT_RATE'].values[0]

        # Imputers

        df[self.continuous_features] = self.imputers['imputer_mean'].transform(df[self.continuous_features])
        df[self.categorical_features] = self.imputers['imputer_most_frequent'].transform(df[self.categorical_features])

        # Label Encoders
        for i in self.list_transformers[:-1]:
            df[i] = self.transformers[i].transform(df[i])

        # Scaler

        df = pd.DataFrame(self.transformers['Scaler'].transform(df), columns = self.all_features)
        
        return df, req_info



    def predict(self, df: pd.DataFrame):

        prediction = self.model.predict(df.values)
        return prediction



    def explain_prediction(self, df: pd.DataFrame):

        expl_details = self.explainer.explain_instance(
            df.values.reshape(-1),
            self.model.predict_proba,
            num_features=6
        )

        return pd.DataFrame(expl_details.as_map()[1], columns=['Feature_idx', 'Scaled_value'])
        