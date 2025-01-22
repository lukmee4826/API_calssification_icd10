from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from prediction import predict_batch, predict_single
import tempfile

app = FastAPI(
    title="ICD-10 Prediction API",
    description="""
This API allows for single or batch predictions of ICD-10 categories using textual input.

**Features**:
- Perform single predictions with a text input.
- Upload a CSV or XLSX file for batch predictions.
""",
    version="1.0.0",
    contact={
        "name": "Your Name",
        "email": "your.email@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Models for input/output
class SinglePredictionOutput(BaseModel):
    input: str
    predicted_1: str
    confident_1: float
    predicted_2: str
    confident_2: float
    predicted_3: str
    confident_3: float

class BatchPredictionExample(BaseModel):
    file: str = "sample_data.csv"
    column_name: Optional[str] = "text"

@app.get("/")
async def root():
    """
    Health check endpoint to verify the API is running.
    """
    return {"message": "Status Ok"}

@app.get("/predict_single/{input}", response_model=SinglePredictionOutput,
         summary="Predict ICD-10 Classification for Single Input",
    description="""
    Perform a single ICD-10 classification based on the input text.
    
    This endpoint accepts a text input (e.g., symptoms or diagnosis) and returns the top 3 ICD-10 categories 
    with their confidence scores.

    **Example Input**:
    - "Patient reports severe headaches and nausea."

    **Example Output**:
    ```json
    {
        "input": "Patient reports severe headaches and nausea",
        "predicted_1": "Certain infectious and parasitic diseases",
        "confident_1": 0.92,
        "predicted_2": "Endocrine, nutritional and metabolic diseases",
        "confident_2": 0.87,
        "predicted_3": "Diseases of the respiratory system",
        "confident_3": 0.76
    }
    ```
    """,)
async def predict_single_text(input: str):
    """
    Perform ICD-10 classification for a single input text and return the top 3 predictions with confidence scores.
    """
    try:
        results = predict_single(input)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch/", response_description="Download the file containing batch predictions.")
async def predict_batch_file(
    file: UploadFile = File(...),
    column_name: str = Query(default="text", description="The name of the column containing the input texts."),
    language: str = Query(default="english", description="Language of the input texts (e.g., 'thai', 'english', or 'both')."),
):
    """
    Perform batch predictions for ICD-10 classification.

    - **file**: Upload a CSV or XLSX file containing input texts.
    - **column_name**: The name of the column in the file containing the texts to classify.
    - **language**: Specify the language of the input texts.

    **Output**:
    - A downloadable CSV file containing predictions for each input text.
    """
    try:
        # Validate file type
        if not (file.filename.endswith(".csv") or file.filename.endswith(".xlsx")):
            raise HTTPException(status_code=400, detail="Uploaded file must be a CSV or XLSX.")

        # Save the uploaded file
        file_contents = await file.read()
        input_data = tempfile.NamedTemporaryFile(delete=False, suffix=".csv" if file.filename.endswith(".csv") else ".xlsx")
        output_data = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")

        with open(input_data.name, "wb") as temp_file:
            temp_file.write(file_contents)

        # Read the file to check if the column exists
        if file.filename.endswith(".csv"):
            df = pd.read_csv(input_data.name)
        else:
            df = pd.read_excel(input_data.name)

        if column_name not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{column_name}' not found in the uploaded file. Available columns: {list(df.columns)}"
            )

        # Call the prediction logic
        predict_batch(input_data.name, output_data.name, column_name, language)  # Pass 'language' to prediction logic

        return FileResponse(output_data.name, media_type="text/csv", filename="predictions.csv")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
