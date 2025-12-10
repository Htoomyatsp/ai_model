from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import keras
import joblib
import sklearn

app = FastAPI()
# Load the model
model = keras.models.load_model('../checkpoint/model.keras')
# Load the scalers
feature_scaler = joblib.load('../checkpoint/feature_scaler.save')
target_scaler = joblib.load('../checkpoint/target_scaler.save')

class InputData(BaseModel):
    data: list[list[float]]  # 20 timesteps x 21 features


@app.post("/predict")
def predict(input: InputData):
    input_array = np.array(input.data)  # shape (20, 21)
    if input_array.shape != (20, 21):
        return {"error": "Expected input shape (20, 21)"}

    # Scale input
    input_scaled = feature_scaler.transform(input_array)  # shape (20, 21)
    input_scaled = input_scaled.reshape(1, 20, 21)  # model expects 3D input

    # Predict
    prediction_scaled = model.predict(input_scaled)

    # Inverse-transform output
    prediction_original = target_scaler.inverse_transform(prediction_scaled)

    return {"prediction": prediction_original.tolist()}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


