#!/bin/bash
MODELO=$1

CARPETA_DE_MODELOS=$(pwd)/modelos
LINK_MODELO_ENCODER="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/$MODELO/$MODELO-encoder/FP32/$MODELO-encoder"
LINK_MODELO_DECODER="https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/$MODELO/$MODELO-decoder/FP32/$MODELO-decoder"

if [ ! -f "$CARPETA_DE_MODELOS/$MODELO/decoder/$MODELO-decoder.xml"  ]; then
    echo "Descargando Modelo.. "
    curl $LINK_MODELO_DECODER.xml \
     --create-dirs -o $CARPETA_DE_MODELOS/$MODELO/decoder/$MODELO-decoder.xml
     curl $LINK_MODELO_DECODER.bin \
     --create-dirs -o $CARPETA_DE_MODELOS/$MODELO/decoder/$MODELO-decoder.bin
     echo "Modelo descargado."
fi

if [ ! -f "$CARPETA_DE_MODELOS/$MODELO/encoder/$MODELO-encoder.xml"  ]; then
    echo "Descargando Modelo.. "
    curl $LINK_MODELO_ENCODER.xml \
     --create-dirs -o $CARPETA_DE_MODELOS/$MODELO/encoder/$MODELO-encoder.xml
     curl $LINK_MODELO_ENCODER.bin \
     --create-dirs -o $CARPETA_DE_MODELOS/$MODELO/encoder/$MODELO-encoder.bin
    echo "Modelo descargado."
fi