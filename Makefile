ENV_NAME=ml_deploy_fast_api
ENV_FILE=environment.yml

update-env:
	conda env update -f $(ENV_FILE) -n $(ENV_NAME)

test:
	pytest  test_model.py