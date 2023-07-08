from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
import sklearn
import joblib
import boto3
import pathlib
from io import StringIO
import argparse
import os
import numpy as np
import pandas as pd

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ =='__main__':
    
    
    print("[INFO] Extracting Arguments")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=0)

    # input data and model directories
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--train-file', type=str, default="train-V-01.csv")
    parser.add_argument('--test-file', type=str, default="test-V-01.csv")

    args, _ = parser.parse_known_args()
    
    print("SKlearn Version: ", sklearn.__version__)
    print("Joblib Version: ", joblib.__version__)
    
    print("[INFO] Reading Data")
    print()
    train_data = pd.read_csv(os.path.join(args.train, args.train_file))
    test_data = pd.read_csv(os.path.join(args.test, args.test_file))
    
    features = list(train_data.columns)
    target = features[-1]
    
    print("Building Training and Testing Datasets")
    print()
    X_train = train_data[features]
    X_test = test_data[features]
    y_train = train_data[target]
    y_test = test_data[target]
    
    print("Column Order: ")
    print(features)
    print()
    
    print("Target Column: ", target)
    print()
    
    print("Data Shape: ")
    print()
    
    print("SHAPE OF TRAINING DATA (85%) --->")
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    print()
    
    
    print("Random Forest Model Building --->")
    print()
    
    model = RandomForestClassifier(n_estimators=args.n_estimators, random_state=args.random_state, verbose=0)
    model.fit(X_train, y_train)
    print()
    
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    
    print("Model is stored at "+model_path)
    print()
    
    y_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_report = classification_report(y_test, y_pred)
    
    print()
    print("Performance Metrics for the Testing Data --->")
    print()
    print("Total rows are: ", X_test.shape[0])
    print("Model Test Accuracy: ", test_accuracy)
    print("Model Test Report: ")
    print(test_report)