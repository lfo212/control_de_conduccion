import cv2
import os
import logging
from abc import abstractmethod
from numpy import ndarray
from openvino.inference_engine import IECore


class Imagen():

    @staticmethod
    def incrementar_area_por_porcentaje(area : dict, porcentaje : int) -> dict:
        return {
            "tl": [max(0, area["tl"][0] * (1 - porcentaje / 100)), max(0, area["tl"][1] * (1 - porcentaje / 100))],
            "br": [min(1, area["br"][0] * (1 + porcentaje / 100)), min(1, area["br"][1] * (1 + porcentaje / 100))]
        }
    
    @staticmethod
    def obtener_posicion_en_enteros(area : dict) -> dict:
        return {
            "tl": (int(area["tl"][0]), int(area["tl"][1])),
            "br": (int(area["br"][0]), int(area["br"][1]))
        }
        
    @staticmethod
    def obtener_posicion_en_imagen_original(area : dict, width : int, height : int) -> dict:
        return {
            "tl": (int(area["tl"][0] * width), int(area["tl"][1] * height)),
            "br": (int(area["br"][0] * width), int(area["br"][1] * height))
        }


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

            _, _, self.height, self.width = _neural_net.input_info[
                self.input_blob
            ].input_data.shape
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
            frame, size=(self.height, self.width), ddepth=cv2.CV_8U
        )
        return self.execution_net.infer(inputs={self.input_blob: blob}).get(
            self.output_blob
        )

class Face:
    def __init__(self, inference_result):
        self.image_id = inference_result[0]
        self.label = int(inference_result[1])
        self.confidence = inference_result[2]
        self.location = {
            "tl": [inference_result[3],inference_result[4]],
            "br": [inference_result[5], inference_result[6]]
            }
    
    def procesar_resultado(self, frame_width, frame_height):

        self.location = Imagen.incrementar_area_por_porcentaje(self.location, 10)
        self.location = Imagen.obtener_posicion_en_imagen_original(self.location, frame_width, frame_height)
        return {
                "tl": self.location["tl"],
                "br": self.location["br"],
                "type": self.label,
                "accuracy": float(self.confidence)
            }

class Detector_de_rostros(Motor_de_inferencia):

    def get_output_blob(self) -> ndarray:
        return next(iter(self.execution_net.outputs))
    
    def procesar_frame(self, frame):
        input_height, input_width, _ = frame.shape
        self.face = Face(super().procesar_frame(frame)[0][0][0])
        if self.face.confidence < self.confidence_threshold:
            self.log.warning(f"Face detection less than {self.confidence_threshold}, accuracy {self.face.confidence}")
            return {}

        if self.face.image_id < 0:
            self.log.warning(f"Invalid image id {self.face.image_id}")
            return {}

        return self.face.procesar_resultado(input_width, input_height) 
        

    