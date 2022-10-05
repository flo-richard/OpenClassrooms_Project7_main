import uvicorn
from fastapi import FastAPI, Request
from model import ScoringModel
import pandas as pd


app = FastAPI()
Model = ScoringModel()

# @app.get('/')
# def index():
#     return {'message': 'Hello World'}

#@app.get('/{name}')
#def get_name(name: str):    
#    return {'message': f'Hello, {name}'}

# if __name__ == '__main__':
#     uvicorn.run(app, hist='127.0.0.1', port=8000)


@app.post('/getPrediction')
async def get_prediction(info : Request):
    req_info = await info.json()
    # Ajouter les fonctions qui font le traitement ici
    # Pour l'instant on renvoit l'info telle qu'on la reçoit

    print(req_info)

    req_info_new = Model.preprocessing(req_info)

    print('req info new: ', req_info_new)
    test_return = {
         "status": "SUCCESS",
         "data": req_info_new
        }
    #plop.update()
    print('test return : ', test_return)
    
    return test_return