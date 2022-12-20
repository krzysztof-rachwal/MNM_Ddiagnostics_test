import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import classification_report

def predict():
    
    MODEL_FILE_XGB = os.environ["MODEL_FILE_XGB"]


    # Load, read and normalize training data
    df = pd.read_csv('dataset.csv', index_col=0)  
    df = df.dropna(subset=['Class'])
    
    
    X_test = df.drop('Class', axis=1)
    y_test = df['Class']
   
    print("Shape of the test data")
    print(X_test.shape)
    print(y_test.shape)
        
    # Run model
    model = load(MODEL_FILE_XGB)
    print(model)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
if __name__ == '__main__':
    predict()