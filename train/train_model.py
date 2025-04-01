# Script to train machine learning model.

import os

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import joblib

#  load in the data.
data = pd.read_csv("data/census_clean.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test,  categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
model = train_model(X_train, y_train)
joblib.dump(model,   os.path.join("../model","model.pkl"))
joblib.dump(encoder, os.path.join("../model", "encoder.pkl"))
joblib.dump(lb,      os.path.join("../model", "lb.pkl"))


fname = ['X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv' ]
for idx, files in enumerate([X_train, y_train, X_test, y_test ]):
    outfile = os.path.join('data/', fname[idx])
    np.savetext(outfile, fname, delimiter=",")

preds = inference(X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)




