download_models:
	sudo chmod 775 -R scripts
	./scripts/descargar_modelo.sh face-detection-retail-0004
	./scripts/descargar_modelo.sh facial-landmarks-98-detection-0001
	./scripts/descargar_modelo.sh head-pose-estimation-adas-0001
	./scripts/descargar_modelo.sh face-reidentification-retail-0095
	./scripts/descargar_modelo_action_recognition.sh driver-action-recognition-adas-0002
	./scripts/descargar_dlib_model.sh
start: 	download_models
	./scripts/start.sh
