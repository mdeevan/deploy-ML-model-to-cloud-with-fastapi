
[![Python application](https://github.com/mdeevan/deploy-ML-model-to-cloud-with-fastapi/actions/workflows/python-app.yml/badge.svg)](https://github.com/mdeevan/deploy-ML-model-to-cloud-with-fastapi/actions/workflows/python-app.yml)


# About
This is a Supervised Machine Learning project, classifying a salary of an invdividual based on the census data. The project utilizes full lifecycle of MLOPS including CI/CD pipeline.

* DVC (Data Version Control):
   - To version control the data, and model revisions with the ability to rollback to previous versions
   - Metrics tracking as version for each of experiments
   - Parametrize experiments, via a yaml file to avoid changing code in experimentation
   - The data files, which are large, are stored in S3, whereas MD5 file signatures are tracked in GIT
     
* Github actions: (continuous integration)
   - As code is pushed into the github repository, code is formatted (black), linted (pylint) and tested (pytest)
   - on successful completion of the build job that include code testing, deploy job deploys to render via github actions
    
* Render: (continous deployment)
   - the checked-in code is deployed automatically, once build, lint and tests are passed

* StreamLit: (Front end)
   - Code is deployed on Streamlit for inteacting with the model and obtaining predictions.

# Interacting/Training the application 
   ### Fastapi
   - FastAPI: https://deploy-ml-model-to-cloud-with-fastapi.onrender.com 
      * the link will be down due to inactivity and takes some time to be reactivated (because of free version),
      * to run locally use uvicorn main:app
      * use following to check if service is active  
         curl -X 'GET'   'https://deploy-ml-model-to-cloud-with-fastapi.onrender.com/'   -H 'accept: application/json'

   ### Streamlit
   - Streamlit application : https://deploy-ml-model-to-cloud-with-fastapi-ewhwkycktnpksn4brtqume.streamlit.app/
      * a free version may result in app being dormant, so will need to be refreshed.
      * to run in locally use streamlit run census_app.py

   ### Running experiments
   - DVC pipeline
      * use the following to run the experiment, modify the n_estimators as needed
         dvc exp run --set-param n_estimators=145
      * all customizaable parameters are defined in params.yml, to avoid need to make changes in code
      * dvc.yml
         - Metrics and outfiles defined in dvc.yaml (pipeline)
         - metrics and outfiles are tracked by dvc and stored in S3 in  this case

# Environment Set up
### Setup development environment

setup environment in one of the following few ways
one can use a cloud ubuntu 2022 machine with about 30 GB or RAM and some medium compute. For this particular project I made use of t3.medium EC2 instance with 30 GB of RAM.

- Setup development environment
   - Using conda  
      follow instructions here to [install miniconda or anaconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer)   

      here is how to install miniconda
      ```
      # download miniconda
      wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

      # install miniconda
      bash ~/Miniconda3-latest-Linux-x86_64.sh
      
      once the installaton is successful. create enviornment. it uses environment.yml file
      make update-env
      ```

   - using python virual env
      ```
      python -m venv venv

      # setup environment with requirement.txt file
      pip install -r requirements.txt

      ```

- for DVC, S3 was used as storage medium

   Follow the details [here to setup aws credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html) 
```
   use aws configure to setup the access and secret keys. Access and secret keys are first to be created and obtained from AWS either through console or via API.
   here are the details on configuring credentials locally

   $ aws configure
   (sample data)
   AWS Access Key ID [None]: AKIAIOSFODNN7EXAMPLE
   AWS Secret Access Key [None]: wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
   Default region name [None]: us-west-2
   Default output format [None]: json

```

### Set up S3

* In your CLI environment install the<a href="https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html" target="_blank"> AWS CLI tool</a>.
* In the navigation bar in the Udacity classroom select **Open AWS Gateway** and then click **Open AWS Console**. You will not need the AWS Access Key ID or Secret Access Key provided here.
* From the Services drop down select S3 and then click Create bucket.
* Give your bucket a name, the rest of the options can remain at their default.

To use your new S3 bucket from the AWS CLI you will need to create an IAM user with the appropriate permissions. The full instructions can be found <a href="https://docs.aws.amazon.com/IAM/latest/UserGuide/id_users_create.html#id_users_create_console" target="_blank">here</a>, what follows is a paraphrasing:

* Sign in to the IAM console <a href="https://console.aws.amazon.com/iam/" target="_blank">here</a> or from the Services drop down on the upper navigation bar.
* In the left navigation bar select **Users**, then choose **Add user**.
* Give the user a name and select **Programmatic access**.
* In the permissions selector, search for S3 and give it **AmazonS3FullAccess**
* Tags are optional and can be skipped.
* After reviewing your choices, click create user. 
* Configure your AWS CLI to use the Access key ID and Secret Access key.

### Make file
Use makefile to lint, format, update environment or run test
  



# Folder structure

| path/file | description|
| ---- | --- |
.  | root
├── EDA_cleanup.ipynb     |  EDA to clean census.csv and save as census_clean.csv    
├── Makefile              | 
├── README.md  | This file
├── census_app.py  | streamlit application
├── census_class.py  | training code as a python class
├── census_class_test.py  | pytests for census_class
├── compute_slices.py  | compute metrics for categorical features
├── data  |
│   ├── X_test.csv  | Encoded test set features (DVC tracked)
│   ├── X_train.csv  | Encoded train set feature (DVC tracked)
│   ├── census.csv  | original downloaded data
│   ├── census_clean.csv  | cleaned version after EDA of census data
│   ├── metrics.json  | metrics produced from experiments  (DVC tracked)
│   ├── slice_output.txt  | metrics output from compute_slices.py
│   ├── slicer.csv  |
│   ├── test.csv  | Features test split of the census_clean.csv (DVC tracked).
│   ├── train.csv  | Features train split of the census_clean.csv (DVC tracked) 
│   ├── y_test.csv  | Binarized test set Target set faetures  (DVC tracked)
│   └── y_train.csv  | Binarized train set Target set faetures  (DVC tracked)
├── dvc.lock  | 
├── dvc.yaml  | dvc pipeline
├── environment.yml   | enviroment setup with conda
├── experiments.txt  | environment setup with python
├── htmlcov  (collapsed)  | coverage output folder
├── logs  | 
│   └── census.log  | log created from experiments
├── main.py  | FAST API 
├── main_test.py  | test main.py
├── ml  |
│   ├── __ init__.py  |
│   ├── data.py  | features coding and target binarization
│   └── model.py  | train, compute metrics and inference, referencd from census_class
├── model   |
│   ├── encoder.pkl  | featured encoder -  (DVC tracked)
│   ├── lb.pkl  | label binarizer -  (DVC tracked)
│   └── model.pkl  | trained model -  (DVC tracked)
├── model_card_template.md | description of the model
├── params.yaml  | customizatable parameters
├── requirements.txt  | environment setup with python
├── sanitycheck.py  |
├── screenshots (collapsed)|  
├── setup.py  |
└── train_model.py  | train the model

# Metrics

### Experiments

![Metrics from experiments](./screenshots/experiments-metrics.png)


# Code Coverage

![](./screenshots/test_coverage.png)

# Test Result
![](./screenshots/test_results.png)