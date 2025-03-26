import pandas as pd
import joblib

import dvc.api
import os


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

    file_path = params['data']['path']
    clean_data_file = params['data']['clean_file']

    test_size = params['data']['test_size']

    model_path = params['model']['model_path']
    model_name = params['model']['model_name']

    cat_features = params['cat_features']

    df = pd.read_csv(os.path.join(file_path, clean_data_file))

    for cat in cat_features:
        # print(feature)

        for feature in df[cat].unique():

            mask = df[cat] == feature
            print(f"{cat}: {feature} - {df[mask].shape[0]}")


    slice_options = df[feature].unique().tolist()
    perf_df = pd.DataFrame(index=slice_options, 
                            columns=['feature','n_samples','precision', 'recall', 'fbeta'])
    for option in slice_options:
        slice_mask = df[feature]==option

        slice_y = y[slice_mask]
        slice_preds = preds[slice_mask]
        precision, recall, fbeta = compute_model_metrics(slice_y, slice_preds)
        
        perf_df.at[option, 'feature'] = feature
        perf_df.at[option, 'n_samples'] = len(slice_y)
        perf_df.at[option, 'precision'] = precision
        perf_df.at[option, 'recall'] = recall
        perf_df.at[option, 'fbeta'] = fbeta

    # reorder columns in performance dataframe
    perf_df.reset_index(names='feature value', inplace=True)
    colList = list(perf_df.columns)
    colList[0], colList[1] =  colList[1], colList[0]
    perf_df = perf_df[colList]

    return perf_df