from abc import abstractmethod
import opencv as cv2
from datetime import datetime

import os
from openvino.inference_engine import IECore


class Motor_de_inferencia:
    def __init__(self, model_xml, model_bin, device, confidence_threshold):
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
    def get_output_blob(self):
        pass

    @abstractmethod
    def procesar_frame(self, frame):
        pass


class Detector_de_rostros(Motor_de_inferencia):
    def get_output_blob(self):
        return next(iter(self.execution_net.outputs))
