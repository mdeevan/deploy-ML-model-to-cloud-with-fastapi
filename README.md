
[![Python application](https://github.com/mdeevan/deploy-ML-model-to-cloud-with-fastapi/actions/workflows/python-app.yml/badge.svg)](https://github.com/mdeevan/deploy-ML-model-to-cloud-with-fastapi/actions/workflows/python-app.yml)


# About
The project is to creating a MLOPS CI/CD pipeline and comprises of following  
* DVC (Data Version Control):
   - To version control the data, and model revisions with the ability to rollback to previous versions
   - Metrics tracking as version for each of experiments
   - Parametrize experiments, via a yaml file to avoid changing code in experimentation  
     
* Github actions: (continuous integration)
   - As code is pushed into the github repository, code is formatted, linted and tested
    
* Render: (continous deployment)
   - the checked-in code is deployed automatically 

* StreamLit: (Front end)
   - Code is deployed on Stramlit for inteacting with the mode, in obtaining the prediction


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

- for DMV, S3 was used as storage medium

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


## API Creation

* Create a RESTful API using FastAPI this must implement:
   * GET on the root giving a welcome message.
   * POST that does model inference.
   * Type hinting must be used.
   * Use a Pydantic model to ingest the body from POST. This model should contain an example.
    * Hint: the data has names with hyphens and Python does not allow those as variable names. Do not modify the column names in the csv and instead use the functionality of FastAPI/Pydantic/etc to deal with this.
* Write 3 unit tests to test the API (one for the GET and two for POST, one that tests each prediction).

## API Deployment

* Create a free Heroku account (for the next steps you can either use the web GUI or download the Heroku CLI).
* Create a new app and have it deployed from your GitHub repository.
   * Enable automatic deployments that only deploy if your continuous integration passes.
   * Hint: think about how paths will differ in your local environment vs. on Heroku.
   * Hint: development in Python is fast! But how fast you can iterate slows down if you rely on your CI/CD to fail before fixing an issue. I like to run flake8 locally before I commit changes.
* Set up DVC on Heroku using the instructions contained in the starter directory.
* Set up access to AWS on Heroku, if using the CLI: `heroku config:set AWS_ACCESS_KEY_ID=xxx AWS_SECRET_ACCESS_KEY=yyy`
* Write a script that uses the requests module to do one POST on your live API.
