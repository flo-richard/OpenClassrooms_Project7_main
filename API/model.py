import os
import pandas as pd
from pydantic import BaseModel
import pickle
import lime
import numpy as np

#PATH = "E:/OpenClassrooms/Projet7/Data"

#class Id(BaseModel):
#    id: int

class ScoringModel:
    def __init__(self):

        # self.df = pd.read_csv('cleaned_data.csv', index_col=0)  # load dataset

        # self.X = self.df.drop('TARGET', axis=1) # split data/target
        # self.y = self.df['TARGET']

        # self.feats = [feat for feat in self.X.columns]

        self.list_features = pickle.load(open('Pickled_objects/features.pkl', 'rb'))['categorical_features_to_encode']
        self.model = pickle.load(open('Pickled_objects/xgb.pkl', 'rb')) # load trained model
        self.transformers = pickle.load(open('Pickled_objects/transformers.pkl', 'rb')) #load transformers
        self.list_transformers = [i for i in self.transformers]  #list of transformers names (columns to be transformed)
        #self.features = pickle.load(open('features.pkl', 'rb')) #load list of features -> à créer dans le notebook

    def preprocessing(self, payload: dict):

        # vérifier si les clés dans self.features sont bien dans payload
        # formater payload au bon format (bonnes clés, avec bonnes valeurs)

        #data = self.X.loc[id]
        print(self.transformers['NAME_CONTRACT_TYPE'].classes_)
        print(payload['NAME_CONTRACT_TYPE'])
        new_payload = {}

        for i in self.list_transformers[:-1]:
            new_payload[i] = self.transformers[i].transform(np.array(payload[i]).reshape(-1))
        #print('new payload : ', new_payload)
        #payload = self.transformers['Scaler'].transform(payload)
        # ajouter imputer
        return new_payload


    def predict(self, id: int):

        if id in self.X.index.to_list():
            

            prediction = self.model.predict(data)

            return prediction

        else:
            print('Error : Id not in database')
            return -1