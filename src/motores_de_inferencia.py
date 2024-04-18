import cv2
import os
import logging
import numpy as np

from openvino.inference_engine import IECore
from objetos import Imagen, Rostro

class Motor_de_inferencia:

    ie_core = IECore()

    def __init__(self, model_xml : str, model_bin : str, device : str, confidence_threshold : float):
        self.log = logging.getLogger("Motor de Inferencia")
        self.confidence_threshold = float(confidence_threshold)
        if not os.path.exists(model_xml):
            raise FileNotFoundError(f"Model xml file missing: {model_xml}")
        if not os.path.exists(model_bin):
            raise FileNotFoundError(f"Model bin file missing: {model_bin}")
        self.log.info("Config reading completed...")
        self.log.info("Confidence = %s", self.confidence_threshold)
        self.log.info("Loading IR files. \n\txml: %s, \n\tbin: %s", model_xml, model_bin)

        # Load OpenVINO model
        _neural_net = self.ie_core.read_network(model=model_xml, weights=model_bin)
        if _neural_net:
            # input_info: A dictionary that maps input layer names to InputInfoPtr objects
            # InputInfoPtr: This class contains information about each input of the network
            self.input_blob_prop = next(iter(_neural_net.input_info))
            _neural_net.batch_size = 1
            self.execution_net = self.ie_core.load_network(
                network=_neural_net, device_name=device.upper()
            )
            self.output_blob_prop = self.get_output_blob_prop()

            self.image_prop = Imagen(*_neural_net.input_info[
                self.input_blob_prop
            ].input_data.shape)
        else:
            self.log.error("Error al cargar red neuronal")

    def get_output_blob_prop(self) -> list:
        return next(iter(self.execution_net.outputs))

    def redimensionar_imagen(self, frame):
        return cv2.dnn.blobFromImage(
            frame, size=(self.image_prop.height, self.image_prop.width), ddepth=cv2.CV_8U
        )

    def procesar_frame(self, frame):
        """[summary]
        :param frame: frame blob
        :type frame: numpy.ndarray
        :rtype: (bool, numpy.ndarray, str)
        """
        self.blob = self.redimensionar_imagen(frame)
        inference_result = self.execution_net.infer(inputs={self.input_blob_prop: self.blob})
        if type(self.output_blob_prop) == list:
            return [inference_result.get(prop) for prop in self.output_blob_prop]
        else:
            return inference_result.get(
                self.output_blob_prop
            )

    #def obtener_resultados()

class Detector_de_rostros(Motor_de_inferencia):
    
    def procesar_frame(self, frame) -> Rostro:
        input_height, input_width, _ = frame.shape
        self.rostro = Rostro.getInstance()
        self.rostro.actualizar_atributos(super().procesar_frame(frame)[0][0][0])
        
        if self.rostro.confidence < self.confidence_threshold:
            self.log.warning(f"Face detection less than {self.confidence_threshold}, accuracy {self.rostro.confidence}")
            self.rostro.rostro_detectado = False
            return self.rostro

        if self.rostro.id < 0:
            self.log.warning(f"Invalid image id {self.rostro.id}")
            self.rostro.rostro_detectado = False
            return self.rostro

        self.rostro.redimensionar_posicion(input_width, input_height)
        return self.rostro