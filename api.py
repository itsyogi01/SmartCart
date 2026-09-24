from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle


# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Load columns
with open("model_columns.pkl", "rb") as file:
    model_columns = pickle.load(file)


app = FastAPI(
    title="SmartCart API",
    description="Customer Purchase Prediction API"
)


class CustomerData(BaseModel):
    data: dict


@app.get("/")
def home():
    return {
        "message": "SmartCart API is running"
    }


@app.post("/predict")
def predict(customer: CustomerData):

    data = pd.DataFrame([customer.data])

    # Convert categorical values
    data = pd.get_dummies(
        data,
        drop_first=True
    )

    # Same columns as training
    data = data.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Prediction
    prediction = model.predict(data)[0]

    probability = model.predict_proba(data)[0][1]

    if prediction == 1:
        result = "Likely to Purchase"
    else:
        result = "Unlikely to Purchase"

    return {
        "prediction": int(prediction),
        "result": result,
        "purchase_probability": round(
            float(probability) * 100,
            2
        )
    }