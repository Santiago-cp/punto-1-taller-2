from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import pandas as pd
import pickle
from typing import Optional
from typing import ClassVar
from sklearn.linear_model import Ridge
from pycaret.regression import setup, compare_models, create_model, tune_model, plot_model, evaluate_model, finalize_model, predict_model, save_model, load_model

# Crear una instancia de FastAPI
app = FastAPI()

# Definir el archivo JSON donde se guardarán las predicciones
file_name = 'predicciones.json'

# Cargar el modelo preentrenado desde el archivo pickle
#model_path = "best_model.pkl"
with open("modelo_ridge.pkl", 'rb') as model_file:
    dt2 = pickle.load(model_file)

prueba = pd.read_csv("prueba_APP.csv",header = 0,sep=";",decimal=",")



class InputData(BaseModel):
    Email: str
    Address: str
    dominio: str
    Tec: str
    Avg_Session_Length: float  
    Time_on_App: float
    Time_on_Website: float
    Length_of_Membership: float

@app.get("/")
def home():
    # Retorna un simple mensaje de texto
    return 'Predicción precio'

# Función para guardar predicciones en un archivo JSON
def save_prediction(prediction_data):
    try:
        with open(file_name, 'r') as file:
            predictions = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        predictions = []

    predictions.append(prediction_data)

    with open(file_name, 'w') as file:
        json.dump(predictions, file, indent=4)

@app.post("/predict")
def predict(data: InputData):
    # Crear DataFrame a partir de los datos de entrada
    user_data = pd.DataFrame([data.dict()])
    user_data.columns = prueba.columns
    user = user_data
    
    # Concatenar los datos del usuario con los datos de prueba
    prueba2 = pd.concat([user,prueba],axis = 0)
    prueba2.index = range(prueba2.shape[0])
    df_test = prueba2.copy()
    predictions = predict_model(dt2, data=df_test)
    predictions["price"] = predictions["prediction_label"]
    prediction_label = predictions.iloc[0]["prediction_label"]

    # Guardar predicción con ID en el archivo JSON
    prediction_result = {"Email": data.Email, "prediction": prediction_label}
    save_prediction(prediction_result)

    return prediction_result

# Ejecutar la aplicación si se llama desde la terminal
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    





