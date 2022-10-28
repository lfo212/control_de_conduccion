import cv2
import os
import logging
from abc import abstractmethod
from numpy import ndarray
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
            self.input_blob = next(iter(_neural_net.input_info))
            _neural_net.batch_size = 1
            self.execution_net = self.ie_core.load_network(
                network=_neural_net, device_name=device.upper()
            )
            self.output_blob = self.get_output_blob()

            self.image_prop = Imagen(*_neural_net.input_info[
                self.input_blob
            ].input_data.shape)
        else:
            self.log.error("Error al cargar red neuronal")

    @abstractmethod
    def get_output_blob(self) -> ndarray:
        pass

    def procesar_frame(self, frame) -> dict:
        """[summary]
        :param frame: frame blob
        :type frame: numpy.ndarray
        :rtype: (bool, numpy.ndarray, str)
        """

        blob = cv2.dnn.blobFromImage(
            frame, size=(self.image_prop.height, self.image_prop.width), ddepth=cv2.CV_8U
        )
        return self.execution_net.infer(inputs={self.input_blob: blob}).get(
            self.output_blob
        )

class Detector_de_rostros(Motor_de_inferencia):

    def get_output_blob(self) -> ndarray:
        return next(iter(self.execution_net.outputs))
    
    def procesar_frame(self, frame):
        input_height, input_width, _ = frame.shape
        self.rostro = Rostro(super().procesar_frame(frame)[0][0][0])
        if self.rostro.confidence < self.confidence_threshold:
            self.log.warning(f"Face detection less than {self.confidence_threshold}, accuracy {self.rostro.confidence}")
            return {}

        if self.rostro.id < 0:
            self.log.warning(f"Invalid image id {self.rostro.id}")
            return {}

        return self.rostro.procesar_resultado(input_width, input_height) 
        
        

    