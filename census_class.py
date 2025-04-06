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

    def __init__(self, nrows= None, inference=False):
        """
        Initialize the class object
        INPUT:
            nrows: number of rows to read, default is all rows in the file
                    during Test, rows are reduced to speed up test execution
            inference: same class object is used for inference as well as
                        training and pytest
                        when inference is TRUE: a single row is passed and we
                            will not split into training test
                        when inference is FALSE: the class serves training and pytest

        """

        #  readin the parameter file and initialize the variables

        # this allows the ability to load fewer rows when running test
        self.nrows = nrows 
        
        # load the parameters from the parameter file
        self.params = dvc.api.params_show()

        self.n_estimators = self.params['n_estimators']

        self.data_path = self.params['data']['path']
        self.clean_data_file = self.params['data']['clean_file']

        #  0 test size would mean effectively no split
        self.test_size = 0 if inference else self.params['data']['test_size'] 

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
    def _read_data(self, path=None, data_file=None):
        """
        INPUTS: 
            path: path of the data file
            data_file: data-filename, allows to use the a single row or
                        multi-rows data for inference
                        Single row used with fastapi predict
                        multi-row used in running test and training
        RETURN:
            none
        """
        data_path = self.data_path if path is None else path
        clean_data_file = self.clean_data_file if data_file is None else data_file

        self.data = pd.read_csv(os.path.join(data_path, clean_data_file), nrows=self.nrows)

    def _split_data(self):
        self.train, self.test = train_test_split(self.data, test_size=self.test_size)

    def _process_data(self, features=None, training_flag:bool=True, encoder=None, lb=None):
        """
        process data calls the process_data method in data library to create the features
        and encode the target

        INPUT:
            features: when none, its assumed to be the training
            training_flag: default is training
            encoder: to one-hot-encode to categories
            lb : for label binarization
        """

        input_features = self.train if features is None else features

        features, target, encoder, lb = process_data(input_features,
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

        # self.X_test.to_csv(os.path.join(data_path, "X_test.csv"), index=False)
        # self.y_test.to_csv(os.path.join(data_path, "y_test.csv"), index=False)

        fname = ['X_train.csv', 'y_train.csv', 'X_test.csv', 'y_test.csv' ]
        for idx, file in enumerate([self.X_train, self.y_train, self.X_test, self.y_test ]):
            outfile = os.path.join(data_path, fname[idx])
            np.savetxt(outfile, file, delimiter=",")


    def make_inference(self, model, features, path=None):
        """
        Make target prediction for the passed data
        which would be a single row when called from API
        and multiple rows when called during training and pytest

        INPUT:
            model: mode used for predictions
            features: features for predictions
            # targets: targets to compare preduction again
            path:
        RETURN:
            preds: resulting predictions are returned 
        """
        if model    is None: model=self.model
        if features is None: features=self.X_test
        # if targets  is None: targets=self.y_test
        data_path = self.data_path if path is None else path


        print(f"\nfeatures: {features}\n")
        preds = inference(model, features)

        return preds

    def _compute_metrics(self, targets, preds, path=None):
        data_path = self.data_path if path is None else path

        precision, recall, fbeta = compute_model_metrics(targets, preds)

        outfile = os.path.join(data_path, self.metric_file)
        print(f"output metric: {outfile}")

        with open(outfile, "w") as f:
            json.dump({'precision': precision, 
                    'recall' : recall, 
                    'fbeta': fbeta}, f)

    def execute_training(self):
        """
        orchesterate the execution of the training, as each step in interdependent,
        the methods are created as private
        """
        self._read_data()
        self._split_data()
        self.X_train, self.y_train, self.encoder, self.lb = self._process_data(training_flag=True)

        self.X_test , self.y_test,  _, _ = self._process_data(training_flag=False, features=self.test, encoder=self.encoder, lb=self.lb)
        self._train_model(self.X_train, self.y_train, self.n_estimators)
        self._save_model()
        self._save_data_split()
        self.preds = self.make_inference(self.model, self.X_test)
        self._compute_metrics(self.y_test, self.preds)

    def execute_inference(self, model, encoder, lb, df:pd.DataFrame=None):
        """
        in case of inference call from the front end for a single row of data
        there is no split, so test will be empty and train will have one row
        here we call the inference with train, having a single row 
        """
        # self._read_data()
        # self._split_data(inference=True)
        features , _,  _, _ = self._process_data(features=df, training_flag=False, encoder=encoder, lb=lb)

        pred = self.make_inference(model, features)
        print(f"census_class : execute inference- prediction = {pred}")

        return pred


# if __name__ == "__main__":
#     census = Census()
#     census.execute()
