# OpenClassrooms_Projet7


This project is an online scoring application to estimate whether a bank client will be able to repay a credit. The decision is made using a Machine Learning (ML) model trained on data that can be found at https://www.kaggle.com/c/home-credit-default-risk/data . 

This repository contains the codes to clean and preprocess the data, as well as the implementation, optimization and training of several ML algorithms to be compared. The model selected here is XGBoost. The code also implements global and local interpretability of the decision made by the model using the LIME library. The trained model is saved as a .pkl file to be used in a REST API, as well as all required transformers so that the model can be directly applied to raw data. The details of the API can be found at https://github.com/flo-richard/scoring_model_api .

This API being non-interactive, it is to be used either via a jupyter notebook (see the api_request.ipynb notebook for an example) or with an interactive dashboard (details at https://github.com/flo-richard/OC_project7_dashboard).

Both the API and the dashboard are deployed as heroku applications :

API : https://scoring-oc7.herokuapp.com/getPrediction

Dashboard : https://credit-score-dashboard-oc7.herokuapp.com/

KEYWORDS : ML 2-classes-classification, Gradient Boosting, Imbalanced data, Custom metric, ML model interpretation