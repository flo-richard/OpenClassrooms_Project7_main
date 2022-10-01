import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()

# @app.get('/')

# def index():
#     return {'message': 'Hello World'}

@app.post('/{getPrediction}')
async def get_name(info : Request):
    req_info = await info.json()
    # Ajouter les fonctions qui font le traitement ici
    # Pour l'instant on renvoit l'info telle qu'on la reçoit
    return {
        "status": "SUCCESS",
        "data": req_info
        }

if __name__ == '__main__':
    uvicorn.run(app, hist='127.0.0.1', port=8000)