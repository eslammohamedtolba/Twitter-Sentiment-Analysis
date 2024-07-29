# Twitter Sentiment Analysis with Machine Learning
This project aims to build a machine learning model that can classify the sentiment of tweets as either positive or negative. 
The model is trained on the Sentiment140 dataset, which is a popular dataset for sentiment analysis tasks. 
The project includes data preprocessing, feature extraction, model training, and evaluation. 
The best-performing model, Logistic Regression, is then deployed as a web application using FastAPI.

![Image about the final project](<Twitter Sentiment Analysis.png>)

## Prerequisites
To run this project, you need to have the following installed on your system:

- Python 3.6 or higher
- FastAPI
- Uvicorn ASGI server
- Joblib
- Pandas
- Matplotlib
- Seaborn
- NLTK
- Scikit-learn
- Kaggle API (for downloading the dataset)

## Overview of the Code
1. Import Dependencies
- pandas for data manipulation
- matplotlib and seaborn for visualization
- nltk for text processing
- scikit-learn for machine learning

2. Load and Inspect Data
- **Loading Data**: The dataset is loaded from a CSV file.
- **Initial Inspection**: Includes checking the shape of the dataset and identifying any missing values.

3. Data Preprocessing
- **Handling Missing Values**: Rows with missing values in the 'text' column are dropped.
- **Column Selection**: Only 'target' and 'text' columns are retained.
- **Label Adjustment**: Replaces label 4 with 1 to simplify binary classification (0 for negative, 1 for positive).
- **Data Balancing**: Visualizes label distribution to check for class balance.

4. Text Processing
- **Stemming**: Applies stemming to the 'text' column using the PorterStemmer.
- **Vectorization**: Converts text data into numerical features using TfidfVectorizer.

5. Model Training and Evaluation
- **Train-Test Split**: The data is split into training (90%) and testing (10%) sets.
- **Model Training**: Models including Logistic Regression and Linear SVC are trained.
- **Model Evaluation**: Each model is evaluated using accuracy, confusion matrix, and classification report. The confusion matrix is visualized for better understanding of model performance.

6. Model Saving and Deployment
- **Save the Best Model**: The best-performing model (Logistic Regression with 78% accuracy) and the TfidfVectorizer are saved using joblib.
- **Deploy with FastAPI**: A FastAPI application is created to provide an interactive web interface for users to input tweets and get predictions. The application loads the saved model and vectorizer, processes the input data, and returns the sentiment prediction (Positive or Negative).


## Model Performance
The Logistic Regression model has achieved an accuracy of 78% on the test data, making it the best performing model for this project.

## Contributions
Contributions to this repository are welcome! Feel free to submit pull requests, report issues, or suggest improvements. Your input will help enhance this project and make it more robust.
