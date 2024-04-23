#!/bin/bash

GREEN='\033[0;32m'
NC='\033[0m'
MODEL_NAME=shape_predictor_68_face_landmarks
MODEL_FOLDER=$(pwd)/modelos/${MODEL_NAME}
MODEL_FILE=${MODEL_NAME}.dat

if [ ! -f "${MODEL_FOLDER}/${MODEL_FILE}" ]; then
	git clone https://github.com/davisking/dlib-models.git
	bunzip2 dlib-models/${MODEL_FILE}.bz2
	mkdir ${MODEL_FOLDER}
	mv dlib-models/${MODEL_FILE} ${MODEL_FOLDER}/
	rm -rf dlib-models
	echo -e "${GREEN}DLIB model downloaded.${NC}"
	else
		echo -e "${GREEN}DLIB already downloaded.${NC}"
	fi
