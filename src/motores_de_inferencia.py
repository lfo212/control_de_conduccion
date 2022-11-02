import cv2
import os
import logging
from abc import abstractmethod
from numpy import ndarray, uint8, fromfile, expand_dims
from openvino.inference_engine import IECore
from objetos import Imagen, Rostro
from imutils import paths
from scipy import spatial

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

    def get_output_blob(self) -> ndarray:
        return next(iter(self.execution_net.outputs))

    def redimensionar_imagen(self, frame):
        return cv2.dnn.blobFromImage(
            frame, size=(self.image_prop.height, self.image_prop.width), ddepth=cv2.CV_8U
        )

    def procesar_frame(self, frame) -> dict:
        """[summary]
        :param frame: frame blob
        :type frame: numpy.ndarray
        :rtype: (bool, numpy.ndarray, str)
        """
        self.blob = self.redimensionar_imagen(frame)
        return self.execution_net.infer(inputs={self.input_blob: self.blob}).get(
            self.output_blob
        )

class Detector_de_rostros(Motor_de_inferencia):
    
    def procesar_frame(self, frame) -> Rostro:
        input_height, input_width, _ = frame.shape
        self.rostro = Rostro.getInstance()
        self.rostro.actualizar_atributos(super().procesar_frame(frame)[0][0][0])
        
        if self.rostro.confidence < self.confidence_threshold:
            self.log.warning(f"Face detection less than {self.confidence_threshold}, accuracy {self.rostro.confidence}")
            self.rostro.nombre = "DESCONOCIDO"
            self.rostro.rostro_detectado = False
            return self.rostro

        if self.rostro.id < 0:
            self.log.warning(f"Invalid image id {self.rostro.id}")
            self.rostro.nombre = "DESCONOCIDO"
            self.rostro.rostro_detectado = False
            return self.rostro

        self.rostro.redimensionar_posicion(input_width, input_height)
        return self.rostro
        
class Identificador_de_rostros(Motor_de_inferencia):

    """
    def redimensionar_imagen(self, frame):
        resized_frame = cv2.resize(frame, (self.image_prop.width, self.image_prop.height))

        # reshape to network input shape
        # Change data layout from HWC to CHW
        return expand_dims(resized_frame.transpose(2, 0, 1), 0)
    """
    def procesar_frame(self, frame) -> list:
        return [x[0][0] for x in list(super().procesar_frame(frame)[0])]

    def generar_base_de_datos_de_choferes(self, directorio_imagenes_choferes : str):
        self.choferes_dict = {}
        for image_path in paths.list_images(directorio_imagenes_choferes):
            name = os.path.basename(image_path).split(".")[0]
            if name.startswith("chofer_"):
                name = name[len("chofer_"):]
            name = name.replace("_", " ")
            try:
                imagen = cv2.imdecode(fromfile(image_path, dtype=uint8), cv2.IMREAD_COLOR)
            except (IOError, cv2.error):
                imagen = None
                self.log.warning(f"Archivo invalido: {image_path}")
            if imagen is not None:
                self.choferes_dict[name] = self.procesar_frame(imagen)
    
    def obtener_nombre_conductor(self, frame) -> str:
        new_vector = self.procesar_frame(frame)
        for name, vector in self.choferes_dict.items():
            result = 1 - spatial.distance.cosine(vector, new_vector)
            if result >= self.confidence_threshold:
                return name
        return "DESCONOCIDO"

class Detector_de_rasgos_faciales(Motor_de_inferencia):

    def sign(self, number : float):
        if number > 0:
            return 1
        elif number < 0:
            return -1
        else:
            return 0

    def procesar_frame(self, face_frame, frame_shape, posicion_inicial):
        frame_height, frame_width, _ = frame_shape
        x_inicial = posicion_inicial[0]
        y_inicial = posicion_inicial[1]
        drf_result = super().procesar_frame(face_frame)[0]
        position_points = []
        rows, colums = drf_result[0].shape
        for point in drf_result:
            _value = -1
            x_position = 0
            y_position = 0
            for x in range(colums):
                for y in range(rows):
                    if point[x][y] > _value:
                        _value = point[x][y]
                        x_position = x
                        y_position = y 
            if _value > 0:
                position_points.append([x_position, y_position])
            else:
                position_points.append([-1,-1])
        for index, coord in enumerate(position_points):
            if 1 < coord[0] and coord[0] < colums - 1 and 1 < coord[1] and coord[1] < rows - 1:
                diffFirst = drf_result[index][coord[0]+1][coord[1]] - drf_result[index][coord[0]-1][coord[1]]
                diffSecond = drf_result[index][coord[0]][coord[1]+1] - drf_result[index][coord[0]][coord[1]-1]
                coord[0] += self.sign(diffFirst) * 0.25
                coord[1] += self.sign(diffSecond) * 0.25
            coord[0] =  y_inicial + int(frame_width * coord[0] / self.image_prop.width)
            coord[1] = x_inicial + int(frame_height * coord[1] / self.image_prop.height)
        return position_points


