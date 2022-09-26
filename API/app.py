import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get('/')

def index():
    return {'message': 'Hello World'}

@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Hello, {name}'}

if __name__ == '__main__':
    uvicorn.run(app, hist='127.0.0.1', port=8000)