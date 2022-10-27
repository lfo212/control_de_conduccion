#!/bin/bash

CARPETA_MODELO=$(pwd)/modelos/facial_landmarks
MODELO=shape_predictor_68_face_landmarks.dat

if [[ ! -f "${CARPETA_MODELO}/${MODELO}" ]]
	then
		git clone https://github.com/davisking/dlib-models.git
		bunzip2 dlib-models/${MODELO}.bz2
		mkdir ${CARPETA_MODELO}
		mv dlib-models/${MODELO} ${CARPETA_MODELO}/
		rm -rf dlib-models
		echo "Modelo DLIB descargado."
	else
		echo "Modelo DLIB ya existe."
	fi