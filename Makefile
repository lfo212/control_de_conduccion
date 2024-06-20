export REPO_FOLDER=$(PWD)
export SCRIPTS= $(REPO_FOLDER)/scripts
export UC_VIRTUAL_ENV=venv
SHELL := /bin/bash

VE=source $(REPO_FOLDER)/$(UC_VIRTUAL_ENV)/bin/activate

install: download_models
	python3 -m venv $(UC_VIRTUAL_ENV) && $(VE) && pip3 install -r requirements-webui.txt
	cd frontend && npm install && npm run build && cd ..
	mkdir -p frontend/public/eventos
	mkdir -p test_files

download_models:
	sudo chmod 775 -R $(SCRIPTS)
	$(SCRIPTS)/descargar_modelo.sh face-detection-retail-0004
	$(SCRIPTS)/descargar_modelo.sh facial-landmarks-98-detection-0001
	$(SCRIPTS)/descargar_modelo.sh head-pose-estimation-adas-0001
	$(SCRIPTS)/descargar_modelo.sh face-reidentification-retail-0095
	$(SCRIPTS)/descargar_modelo_action_recognition.sh driver-action-recognition-adas-0002
	$(SCRIPTS)/descargar_dlib_model.sh
	sudo chmod 775 -R modelos
start:
	-docker-compose -f docker-compose-app.yaml down
	docker-compose -f docker-compose-app.yaml build control_de_manejo
	docker-compose -f docker-compose-app.yaml up -d control_de_manejo
stop:
	docker-compose -f docker-compose-app.yaml stop
stop-api:
	docker-compose -f docker-compose-api.yaml down --remove-orphans
webui:
	docker-compose -f docker-compose-app.yaml down --remove-orphans
	docker-compose -f docker-compose-api.yaml build drivers_api
	docker-compose -f docker-compose-api.yaml up -d drivers_api
	@xhost + > /dev/null
	@# Activate virtual env and Launch WebUI
	@sudo $(SHELL) -c '$(VE) && python3 backend/app.py'
