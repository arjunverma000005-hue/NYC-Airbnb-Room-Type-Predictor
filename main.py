
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Create the FastAPI application
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = joblib.load("Model_Pipeline.pkl")

# Define the column names
columns = [
    "latitude",
    "longitude",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "neighbourhood_group",
    "neighbourhood"
]


# Input validation model
class Features(BaseModel):
    latitude: float = Field( ..., ge=-90, le=90, description="Latitude of the house")
    longitude: float = Field( ..., ge=-180, le=180, description="Longitude of the house")
    price: float = Field(..., gt=0, description="Price of the house" )
    minimum_nights: int = Field(..., ge=1, le=365, description="Minimum number of nights" )
    number_of_reviews: int = Field(..., ge=0,description="Number of reviews")
    reviews_per_month: float = Field(..., ge=0,description="Reviews per month")
    calculated_host_listings_count: int = Field(..., ge=1,description="Number of listings by the host")
    availability_365: int = Field(..., ge=0, le=365,description="Availability throughout the year")
    neighbourhood_group: str = Field(...,min_length=1,description="Neighbourhood group")
    neighbourhood: str = Field(...,min_length=1,description="Neighbourhood")


@app.get("/")
def home():
    return {
        "message": "Welcome to the Room Type Classifier API."
    }


@app.post("/predict")
def predict(features: Features):

    row = pd.DataFrame(
        [features.model_dump()],
        columns=columns
    )

    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0].tolist()

    return {
        "Predicted_room_type": prediction,
        "Probabilities": probabilities}

from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, Field
import joblib
from fastapi.middleware.cors import CORSMiddleware

# Create the FastAPI application
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = joblib.load("Model_Pipeline.pkl")

# Define the column names
columns = [
    "latitude",
    "longitude",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "neighbourhood_group",
    "neighbourhood"
]


# Input validation model
class Features(BaseModel):
    latitude: float = Field( ..., ge=-90, le=90, description="Latitude of the house")
    longitude: float = Field( ..., ge=-180, le=180, description="Longitude of the house")
    price: float = Field(..., gt=0, description="Price of the house" )
    minimum_nights: int = Field(..., ge=1, le=365, description="Minimum number of nights" )
    number_of_reviews: int = Field(..., ge=0,description="Number of reviews")
    reviews_per_month: float = Field(..., ge=0,description="Reviews per month")
    calculated_host_listings_count: int = Field(..., ge=1,description="Number of listings by the host")
    availability_365: int = Field(..., ge=0, le=365,description="Availability throughout the year")
    neighbourhood_group: str = Field(...,min_length=1,description="Neighbourhood group")
    neighbourhood: str = Field(...,min_length=1,description="Neighbourhood")


@app.get("/")
def home():
    return {
        "message": "Welcome to the Room Type Classifier API."
    }


@app.post("/predict")
def predict(features: Features):

    row = pd.DataFrame(
        [features.model_dump()],
        columns=columns
    )

    prediction = model.predict(row)[0]
    probabilities = model.predict_proba(row)[0].tolist()

    return {
        "Predicted_room_type": prediction,
        "Probability": probabilities

    }