from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from nltk.stem.porter import PorterStemmer
from PrepareModel.twitter_sentiment_classifier import stemming
import uvicorn
import joblib

# paths for model and vectorizer
model_path = 'PrepareModel\logisticregressor.sav'
vectorizer_path = 'PrepareModel\\vectorizer.sav'
# Load model and vectorizer
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
stemmer = PorterStemmer()

# Create application
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name = "static")
templates = Jinja2Templates(directory="templates")

# Route for the home page
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Route for the predict page
@app.post("/predict")
async def predict(request: Request, Text: str = Form(...)):
        # Stemming content
    stemmed_Text = stemming(Text, stemmer)
    # Vectorize data
    vectorized_text = vectorizer.transform([stemmed_Text])

    # Make prediction using the loaded model
    prediction = model.predict(vectorized_text)[0]
    print("prediction", prediction)
    if prediction == 1:
        prediction = "Positive"
    else:
        prediction = "Negative"

    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
