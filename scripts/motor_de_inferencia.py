import cv2
import os

from abc import abstractmethod
from numpy import ndarray
from openvino.inference_engine import IECore


class Motor_de_inferencia:
    def __init__(self, model_xml : str, model_bin : str, device : str, confidence_threshold : float):
        # self.log = logging.getLogger("FACE_DETECTION")
        self.model_xml = model_xml
        self.model_bin = model_bin
        self.device = device
        self.confidence_threshold = float(confidence_threshold)
        if not os.path.exists(self.model_xml):
            raise FileNotFoundError(f"Model xml file missing: {self.model_xml}")
        if not os.path.exists(self.model_bin):
            raise FileNotFoundError(f"Model bin file missing: {self.model_bin}")
        # self.log.info("Config reading completed...")
        ##elf.log.info("Confidence = %s", self.confidence_threshold)
        # self.log.info("Loading IR files. \n\txml: %s, \n\tbin: %s", self.model_xml, self.model_bin)

        # Load OpenVINO model
        self.ie_core = IECore()
        self.neural_net = self.ie_core.read_network(model=model_xml, weights=model_bin)
        if self.neural_net:
            self.input_blob = next(iter(self.neural_net.input_info))
            self.neural_net.batch_size = 1
            self.execution_net = self.ie_core.load_network(
                network=self.neural_net, device_name=device.upper()
            )
            self.output_blob = self.get_output_blob()

        self.max_proposal_count = 1
        _, _, self.height, self.width = self.neural_net.input_info[
            self.input_blob
        ].input_data.shape

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


class Detector_de_rostros(Motor_de_inferencia):

    def get_output_blob(self) -> ndarray:
        return next(iter(self.execution_net.outputs))
    
    def set_detection_attributes(self, result):
        self.image_id = result[0]
        self.label = int(result[1])
        self.confidence = result[2]
        self.location = {"x": result[3], "y": result[4]}
        self.size = {"width": result[5], "height": result[6]}
    
    def procesar_resultado(self, width, height):
        self.location["x"] *= width
        self.location["y"] *= height
        self.size["width"] = self.size["width"] * width - self.location["x"]
        self.size["height"] = self.size["height"] * height - self.location["y"]

        bb_width = self.size["width"]
        bb_height = self.size["height"]
        bb_center_x = self.location["x"] + bb_width / 2
        bb_center_y = self.location["y"] + bb_height / 2
        max_size = max(bb_height, bb_width)
        bb_new_width = 1.2 * max_size
        bb_new_height = 1.2 * max_size
        self.location["x"] = int(bb_center_x - bb_new_width / 2)
        self.location["y"] = int(bb_center_y - bb_new_height / 2)
        self.size["width"] = int(bb_center_x + bb_new_width / 2)
        self.size["height"] = int(bb_center_y + bb_new_height / 2)

    def procesar_frame(self, frame):
        input_height, input_width, _ = frame.shape
        results = super().procesar_frame(frame)[0][0][0]
        self.set_detection_attributes(results)
        if self.confidence < self.confidence_threshold:
            #self.log.debug(f"Face detection less than {self.confidence_threshold}, accuracy {result.confidence}")
            return {}

        if self.image_id < 0:
            #self.log.debug(f"Invalid image id {result.image_id}")
            return {}

        self.procesar_resultado(input_width, input_height)

        return {
                "tl": (self.location["x"], self.location["y"]),
                "br": (self.size["width"], self.size["height"]),
                "type": self.label,
                "accuracy": float(self.confidence)
            }
        

    