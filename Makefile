ENV_NAME=ml_deploy_fastapi
ENV_FILE=environment.yml

update-env:
	conda env update -f $(ENV_FILE) -n $(ENV_NAME)

test:
	pytest  test_model.py

format:
	black *.py

lint:
	pylint *.py