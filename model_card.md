# Model Card



## Model Details
    - Person : Muhammad Naveed
    – Model date : April 2025
    – Model version
    – Model type : Random Forest - Supervised Machine Learning
        Its an ensemble model, uses multiple decision trees and combines the result in final classification
    - Objective:
        Train the model on the US census data, and classify whether the individual makes over or under 50K salary per year.
        Among multiple available parameters, *n_estimators* was parameterized in training the model for this project.
        Model was trained multiple times with varying number of n_estimators to evaluate the effectiveness of one model over other.


– Information about training algorithms, parameters, fairness constraints or other applied approaches, and features 
– Paper or other resource for more information – Citation details
– License
– Where to send questions or comments about the model

## Intended Use
    - Its mostly for academic and experiential reason to practice the end to end lifecycle of the MLOPS.
        from training, tuning, and deploying in production, with the emphasis on CI/CD, continuous integration and continuous deployment. 

        While, it would have certainly be helpful to tune multiple parameters in achieving the best model. The objective and focus of this
        project was to make use of the 
            * DVC (data version control) : to version the model, and the data
            * fastapi : To create and deploy publically accessible API  (used render.com for this purpose)
            * streamlit : Frontend to help interact with the model

## Training Data
    The [census data](https://archive.ics.uci.edu/dataset/20/census+income) is downloaded from [here](https://archive.ics.uci.edu/ml/datasets/census+income)


### Features
|Attribute Name|Type|Demographic|
| ---- | ---- | ---- |
|age|Integer|Age|
|workclass|Categorical|Income|
|fnlwgt|Integer|
|education|Categorical|Education Level|
|education-num|Integer|Education Level|
|marital-status|Categorical|Other|
|occupation|Categorical|Other|
|relationship|Categorical|Other|
|race|Categorical|Race|
|sex|Binary|Sex|
|capital-gain|Integer|
|capital-loss|Integer|
|hours-per-week|Integer|
|native-country|Categorical|Other|

### Target
|Attribute Name|Type|Demographic|
| ---- | ---- | ---- |
|income|Binary|Income|

target (income) is stored as <=50K or >50K  
    
    
    
    
### sample data
|age|workclass|fnlgt|education|education-num|marital-status|occupation|relationship|race|sex|capital-gain|capital-loss|hours-per-week|native-country|salary|
| --- | --- | ---- | ---- | ----- | ----- | ---- | ---- | --- | --- | --- | ---- | ---- | ---- | ---- |
|27|Private|257302|Assoc-acdm|12|Married-civ-spouse|Tech-support|Wife|White|Female|0|0|38|United-States|<=50K|
|40|Private|154374|HS-grad|9|Married-civ-spouse|Machine-op-inspct|Husband|White|Male|0|0|40|United-States|>50K|
|58|Private|151910|HS-grad|9|Widowed|Adm-clerical|Unmarried|White|Female|0|0|40|United-States|<=50K|
|22|Private|201490|HS-grad|9|Never-married|Adm-clerical|Own-child|White|Male|0|0|20|United-States|<=50K|
|52|Self-emp-inc|287927|HS-grad|9|Married-civ-spouse|Exec-managerial|Wife|White|Female|15024|0|40|United-States|>50K|



## Evaluation Data  
    - Data was cleaned and then split into Training and Validation Sets as 80-20.
    - some of the rows contained "?" as data, and such rows were removed from the cleaned version
    - Categorical columns were encoded with one-hot-coding
    - label was encoded with label-binarizer

#### Categorical Features
- workclass
- education
- marital-status
- occupation
- relationship
- race
- sex
- native-country

## Metrics  
Precison, Recall and fbeta were used as evaluation metrics

Metrics from running the experiment on varying number of n_estimators

| Experiment                | Created        |precision  |  recall   |  fbeta  | n_estimators|
| ---- | ----        | ----   |  ----   |  ----  | ---- |
| workspace                 | -              |  0.74846  | 0.65258   |0.69724  | 145         |
| main                      | Apr 15, 2025   |  0.72478  | 0.64108   |0.68036  | 133         |
| ├── 1fbd590 [burly-hulk]  | 07:38 PM       |  0.74846  | 0.65258   |0.69724  | 145         |
| ├── 0a2a75b [mousy-peba]  | 07:32 PM       |  0.71272  | 0.62707   |0.66716  | 139         |
| └── 3ba5bd8 [manly-moxa]  | 07:31 PM       |  0.73933  | 0.62656   |0.67829  | 113         |



## Ethical Considerations
When dealing with the peronally identifyable information (PII), masking or anonymization PII data is recommended.


## Caveats and Recommendations

The data is old and is primary used for learning and experimentation. This project focus was on MLOPS CI/CD, and the data suffices the need


## Citation
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf