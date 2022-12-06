# OpenClassrooms_Project7_main


This project is an online scoring application to estimate whether a bank client will be able to repay a credit. The decision is made using a Machine Learning (ML) model trained on data that can be found at https://www.kaggle.com/c/home-credit-default-risk/data . 

This repository contains the codes to clean and preprocess the data, as well as the implementation, optimization and training of several ML algorithms to be compared, with an optimization of the decision threshold with a f-beta function. The model selected here is XGBoost. The code also implements global and local interpretability of the decision made by the model using the LIME library. The trained model is saved as a .pkl file to be used in a REST API, as well as all required transformers so that the model can be directly applied to raw data. The details of the API can be found at https://github.com/flo-richard/OpenClassrooms_project7_api .

This API being non-interactive, it is to be used either via a jupyter notebook (see the api_request.ipynb notebook for an example) or with an interactive dashboard (details at https://github.com/flo-richard/OpenClassrooms_project7_dashboard).

KEYWORDS : ML 2-classes-classification, Gradient Boosting, Imbalanced data, Custom metric, ML model interpretation, decision threshold
