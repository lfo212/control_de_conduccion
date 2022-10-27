#!/bin/bash
MODELO=$1

CARPETA_DE_MODELOS=$(pwd)/modelos
LINK_MODELO="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2021.4/models_bin/3"

if [ ! -f "$CARPETA_DE_MODELOS/$MODELO.xml"  ]; then
    echo "Descargando Modelo.. "
    curl $LINK_MODELO/$MODELO/FP32/$MODELO.xml \
     --create-dirs -o $CARPETA_DE_MODELOS/$MODELO/$MODELO.xml
fi

if [ ! -f "$CARPETA_DE_MODELOS/$MODELO.bin"  ]; then
    curl $LINK_MODELO/$MODELO/FP32/$MODELO.bin \
     --create-dirs -o $CARPETA_DE_MODELOS/$MODELO/$MODELO.bin
    echo "Modelo Descargado"
fi