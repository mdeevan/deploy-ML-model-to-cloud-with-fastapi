# Script to train machine learning model.

import os

from sklearn.model_selection import train_test_split

import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import joblib

import dvc.api
import os
import json

#  load in the data.

params = dvc.api.params_show()

n_estimators = params['n_estimators']

file_path = params['data']['path']
clean_data_file = params['data']['clean_file']

test_size = params['data']['test_size']

model_path = params['model']['model_path']
model_name = params['model']['model_name']
encoder_name = params['model']['encoder_name']
lb_name = params['model']['lb_name']
metric_file = params['model']['metric_file']

# data = pd.read_csv("data/census_clean.csv")
# data = pd.read_csv(os.path.join(file_path, file_name))
data = pd.read_csv(clean_data_file)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=test_size)

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
model = train_model(X_train, y_train, n_estimators)
print(f"path {os.path.join(model_path, model_name)}")
joblib.dump(model,   os.path.join(model_path, model_name))
joblib.dump(encoder, os.path.join(model_path, encoder_name))
joblib.dump(lb,      os.path.join(model_path, lb_name))


preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)

outfile = os.path.join(model_path, metric_file)
print(f"output metric: {outfile}")

with open(metric_file, "w") as f:
    json.dump({'precision': precision, 
               'recall' : recall, 
               'fbeta': fbeta}, f)