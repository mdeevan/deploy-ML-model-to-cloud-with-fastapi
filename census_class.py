# Script to train machine learning model.

import os

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
import joblib

import dvc.api
import os
import json

class Census():

    def __init__(self, nrows= None):

        #  readin the parameter file and initialize the variables

        # this allows the ability to load fewer rows when running test
        self.nrows = nrows 
        
        # load the parameters from the parameter file
        self.params = dvc.api.params_show()

        self.n_estimators = self.params['n_estimators']

        self.data_path = self.params['data']['path']
        self.clean_data_file = self.params['data']['clean_file']

        self.test_size = self.params['data']['test_size']

        self.model_path = self.params['model']['model_path']
        self.model_name = self.params['model']['model_name']
        self.encoder_name = self.params['model']['encoder_name']
        self.lb_name = self.params['model']['lb_name']
        self.metric_file = self.params['model']['metric_file']

        self.cat_features = self.params['cat_features']

        self.data = None
        self.train = None
        self.test = None

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.encoder = None
        self.lb = None

        self.model = None

    # Private method
    def _read_data(self, path=None):
        data_path = self.data_path if path is None else path
        self.data = pd.read_csv(os.path.join(data_path, self.clean_data_file), nrows=self.nrows)

    def _split_data(self):
        self.train, self.test = train_test_split(self.data, test_size=self.test_size)

    def _process_data(self, training_flag:bool=True, encoder=None, lb=None):
        features, target, encoder, lb = process_data(self.train,
                                                     categorical_features=self.cat_features,
                                                     label="salary", 
                                                     training=training_flag,
                                                     encoder=encoder, 
                                                     lb=lb)
        return features, target, encoder, lb

    def _train_model(self, features, target, n_estimators):
        self.model = train_model(features, target, n_estimators)

    def _save_model(self, path=None):

        model_path = self.model_path if path is None else path
        
        try:
            print(f"path {os.path.join( model_path, self.model_name)}")
            joblib.dump(self.model,   os.path.join(model_path, self.model_name))
            joblib.dump(self.encoder, os.path.join(model_path, self.encoder_name))
            joblib.dump(self.lb,      os.path.join(model_path, self.lb_name))
        except:
            print("Error saving model")

    def _save_data_split(self, path=None):

        data_path = self.data_path if path is None else path

        self.train.to_csv(os.path.join(data_path, "train.csv"), index=False)
        self.test.to_csv(os.path.join(data_path, "test.csv"), index=False)

        self.X_test.to_csv(os.path.join(data_path, "X_test.csv"), index=False)
        self.y_test.to_csv(os.path.join(data_path, "y_test.csv"), index=False)


    def make_inference(self, model, features, targets, path=None):

        if model    is None: model=self.model
        if features is None: features=self.X_test
        if targets  is None: targets=self.y_test
        data_path = self.data_path if path is None else path

        preds = inference(model, features)
        precision, recall, fbeta = compute_model_metrics(targets, preds)

        outfile = os.path.join(data_path, self.metric_file)
        print(f"output metric: {outfile}")

        with open(outfile, "w") as f:
            json.dump({'precision': precision, 
                    'recall' : recall, 
                    'fbeta': fbeta}, f)

    def execute(self):
        self._read_data()
        self._split_data()
        self.X_train, self.y_train, self.encoder, self.lb = self._process_data(True)
        self.X_test , self.y_test,  _, _ = self._process_data(False, self.encoder, self.lb)
        self._train_model(self.X_train, self.y_train, self.n_estimators)
        self._save_model()
        self._save_data_split()
        self.make_inference(self.model, self.X_test, self.y_test)


# if __name__ == "__main__":
#     census = Census()
#     census.execute()
