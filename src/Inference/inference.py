from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Initialize the FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define the request model
class PredictionRequest(BaseModel):
    homeTeamName: str
    awayTeamName: str
    matchDate: str
    league: str

# Define the inference endpoint
@app.post("/inference")
async def retreive_model_inference_result(request: PredictionRequest):
    try:
        # Log input data
        logging.info("Received prediction request: %s", request.dict())

        # Placeholder for custom model inference logic
        # Replace with actual model inference code
        # prediction = make_prediction(request.homeTeamName, request.awayTeamName, request.matchDate, request.league)
        # return {
        #     'choice': prediction.choice,
        #     'probability': prediction.probability
        # }
        return {
            'choice': 'HomeTeam',
            'probability': 0.2
        }
    except Exception as e:
        logging.error(f"Error inferencing models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")
