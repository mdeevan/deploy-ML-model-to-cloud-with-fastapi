import pandas as pd
import joblib
from ml.model import inference, compute_model_metrics
from ml.data import process_data

import dvc.api
import os
import csv


def compute_slices():
    """
    Com
    Compute the performance on slices for a given categorical feature
    a slice corresponds to one value option of the categorical feature analyzed
    ------

    Returns
    ------
    Dataframe with
        n_samples: integer - number of data samples in the slice
        precision : float
        recall : float
        fbeta : float
    row corresponding to each of the unique values taken by the feature (slice)
    """    

    params = dvc.api.params_show()

    n_estimators = params['n_estimators']

    data_path = params['data']['path']
    clean_data_file = params['data']['clean_file']
    sliced_name = params['data']['sliced_name']

    test_size = params['data']['test_size']

    model_path = params['model']['model_path']
    model_name = params['model']['model_name']
    encoder_name = params['model']['encoder_name']
    lb_name = params['model']['lb_name']

    cat_features = params['cat_features']

    model = joblib.load(os.path.join(model_path, model_name))
    encoder = joblib.load(os.path.join(model_path, encoder_name))
    lb = joblib.load(os.path.join(model_path, lb_name))


    # X_test = pd.read_csv(os.path.join())
    # y_test = pd.read_csv(os.path.join())
    # df = pd.read_csv(os.path.join(file_path, clean_data_file))
    df_test = pd.read_csv(os.path.join(data_path, "test.csv"))

    slices_inference_list = []
    for cat in cat_features:
        # print(feature)

        for feature in df_test[cat].unique():

            mask = df_test[cat] == feature
            # print(f"{cat}: {feature} - {df_test[mask].shape[0]}")

            X_test, y_test, encoder, lb = process_data(
                df_test[mask],  categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
            )

            preds = inference(model, X_test)
            precision, recall, fbeta = compute_model_metrics(y_test, preds)

            slices_inference_list.append([cat, feature, precision, recall, fbeta])
    
    outfile = os.path.join(data_path, sliced_name)
    with open(outfile, "wt") as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerow(["category", "features", "precision", "recall", "fbeta"])  # write header
        writer.writerows(slices_inference_list)


if __name__ == "__main__":
    compute_slices()