import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import subprocess
import joblib
import shutil
import nltk
import os
nltk.download('stopwords')

def setup_and_download_kaggle_data(destination_folder, extracted_name):
    # Install Kaggle
    subprocess.run(['pip', 'install', 'kaggle'])
    
    # Configure Kaggle
    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
    subprocess.run(['cp', 'kaggle.json', os.path.expanduser('~/.kaggle/')])
    os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
    
    # Download dataset
    os.makedirs(destination_folder, exist_ok=True)
    subprocess.run(['kaggle', 'datasets', 'download', '-d', 'kazanova/sentiment140', '-p', destination_folder])
    
    zip_path = os.path.join(destination_folder, 'sentiment140.zip')
    extracted_path = os.path.join(destination_folder, 'sentiment140')
    
    subprocess.run(['unzip', '-o', zip_path, '-d', extracted_path])
    
    # Find the downloaded CSV file and rename it
    for file in os.listdir(extracted_path):
        if file.endswith('.csv'):
            original_file_path = os.path.join(extracted_path, file)
            new_file_path = os.path.join(destination_folder, extracted_name + '.csv')
            shutil.move(original_file_path, new_file_path)
            break
    
    # Cleanup: remove the zip file and the extracted directory
    os.remove(zip_path)
    shutil.rmtree(extracted_path)

# --------------------------------------------------------- Load data

def load_data(path, encod):
    df = pd.read_csv(path, encoding = encod, header = None)
    df.head()
    # Show shape of the dataset
    print(df.shape)
    # Show some information about dataset
    print(df.info())
    # Show columns of the dataset to know it and know its datatypes
    print(df.columns)

    return df

# ---------------------------------------------------------- Preprocessing data

def preprocess(df):
    # Take copy from original dataset to make preprocessing steps on
    df_processed = deepcopy(df)

    # Change columns names
    df_processed = df_processed.rename(columns={0: 'target', 1: 'ids',2:'date',3:'flag',4:'user',5:'text'})
    df_processed.head()

    # Check about null values to check if we will handle missing values if exist
    df_processed.isnull().sum()

    # Handle invalid data

    # Get the most relevant columns to classify based on that is text and also the output labels
    df_processed = df_processed[['target', 'text']]
    df_processed.head()

    # Check about degree of balancing for the labels counts
    counts = df_processed['target'].value_counts()
    print(counts)
    plt.bar(counts.index, counts)
    plt.show()

    # Replace 4's in the text column into 1's to make the values (0 -> negative, 1 -> positive)
    df_processed['target'] = df_processed['target'].apply(lambda x: 1 if x == 4 else 0)
    df_processed.head()

    # Check the change of replacing the 4's values
    counts = df_processed['target'].value_counts()
    print(counts)
    plt.bar(counts.index, counts)
    plt.show()

    return df_processed


# Create function to take the content and stemmer and stem it
def stemming(content, stemmer):
    # Remove non-alphabetical characters
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    
    # Convert sentences into lowercase sentences and split each into words
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    
    # Prepare stop words
    stop_words = set(stopwords.words('english'))
    # Stem words and remove stop words
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if word not in stop_words]
    stemmed_content = ' '.join(stemmed_content)
    
    return stemmed_content


# Function to perform the model functionalities
def fit_predict(classifier, x_train, y_train, x_test, y_test):
    
    # Make model fit data
    classifier.fit(x_train, y_train)
    
    # Get Score on train and test data
    train_score = classifier.score(x_train, y_train)
    test_score = classifier.score(x_test, y_test)
    print(f'train score is {train_score}, test score is {test_score}')
    
    # Make model predict on test data
    test_prediction = classifier.predict(x_test)
    # Get accuracy, confusion matrix and classification report
    accuracy = accuracy_score(y_test, test_prediction) 
    cf_matrix = confusion_matrix(y_test, test_prediction)
    cl_report = classification_report(y_test, test_prediction)
    
    # Plot confusion matrix
    plt.figure(figsize=(7,7))
    sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Show accuracy and classification report
    print(f'accuracy: {accuracy}\nclassification report:\n{cl_report}', '\n\n')
    
    return classifier




if __name__ == "__main__":
    
    destination_folder = 'PrepareModel'
    extracted_name = 'sentimentanalysis'
    setup_and_download_kaggle_data(destination_folder, extracted_name)

    data_path = os.path.join(destination_folder, extracted_name + '.csv')
    encoding = 'ISO-8859-1'
    
    df = load_data(data_path, encoding)
    df_processed = preprocess(df)
    
    # Show first rows of dataset before stemming it
    df_processed.head()
    # Create stemmer
    stemmer = PorterStemmer()
    # Apply stemming function to the 'text' column in DataFrame
    df_processed['text'] = df_processed['text'].apply(lambda x: stemming(x, stemmer))
    # Show first rows of dataset after stemming it
    df_processed.head()

    # Split data into input and label data
    X = df_processed['text']
    Y = df_processed['target']
    print(f'X shape {X.shape}')
    print(f'Y shape {Y.shape}')
    # Split data into train and test data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.9, stratify = Y, random_state = 42)
    print(f'x train shape {x_train.shape}, x test shape {x_test.shape}')
    print(f'y train shape {y_train.shape}, y test shape {y_test.shape}')

    # Show train data before vectorization
    x_train.head()  
    # Create vectorizer 
    vectorizer = TfidfVectorizer()
    # Vectorize text column
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    # Show train data before vectorization
    print(x_train_vec)

    # Put all required models that can fit this problem
    models = {
        'LogisticRegression': LogisticRegression(),
        'LinearSVC': LinearSVC()
    }
    trained_models = {}
    for name, model in models.items():
        print(f'{name}:\n')
        model = fit_predict(model, x_train_vec, y_train, x_test_vec, y_test)
        trained_models[name] = deepcopy(model)
    
    # Save model and vectorizer
    model_path = 'PrepareModel\logisticregressor.sav'
    vectorizer_path = 'PrepareModel\\vectorizer.sav'

    # Access logistic regression model
    logisticregressor = trained_models['LogisticRegression']

    # Save the Logistic Regression model if it does not already exist
    if not os.path.exists(model_path):
        joblib.dump(trained_models['LogisticRegression'], model_path)
        print(f"Saved Logistic Regression model to {model_path}")
    else:
        print(f"Logistic Regression model already exists at {model_path}")

    # Save the vectorizer if it does not already exist
    if not os.path.exists(vectorizer_path):
        joblib.dump(vectorizer, vectorizer_path)
        print(f"Saved vectorizer to {vectorizer_path}")
    else:
        print(f"Vectorizer already exists at {vectorizer_path}")

