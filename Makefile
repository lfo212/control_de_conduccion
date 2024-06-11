export REPO_FOLDER=$(PWD)
export SCRIPTS= $(REPO_FOLDER)/scripts
export UC_VIRTUAL_ENV=venv
SHELL := /bin/bash

VE=source $(REPO_FOLDER)/$(UC_VIRTUAL_ENV)/bin/activate

download_models:
	sudo chmod 775 -R $(SCRIPTS)
	$(SCRIPTS)/descargar_modelo.sh face-detection-retail-0004
	$(SCRIPTS)/descargar_modelo.sh facial-landmarks-98-detection-0001
	$(SCRIPTS)/descargar_modelo.sh head-pose-estimation-adas-0001
	$(SCRIPTS)/descargar_modelo.sh face-reidentification-retail-0095
	$(SCRIPTS)/descargar_modelo_action_recognition.sh driver-action-recognition-adas-0002
	$(SCRIPTS)/descargar_dlib_model.sh
start: 	download_models
	docker-compose down control_de_manejo
	docker-compose build control_de_manejo
	docker-compose up -d control_de_manejo
stop:
	docker-compose down control_de_manejo
stop-api:
	docker-compose down drivers_api
webui:
	docker-compose build drivers_api
	docker-compose up -d drivers_api
	@xhost + > /dev/null
	@# Activate virtual env and Launch WebUI
	@sudo $(SHELL) -c '$(VE) && python3 backend/app.py'
