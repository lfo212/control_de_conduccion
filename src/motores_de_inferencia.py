import cv2
import os
import logging
import numpy as np

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
                imagen = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            except (IOError, cv2.error):
                imagen = None
                self.log.warning(f"Archivo invalido: {image_path}")
            if imagen is not None:
                self.choferes_dict[name] = self.procesar_frame(imagen)
    
    def obtener_nombre_conductor(self, frame):
        self.rostro = Rostro.getInstance()
        if self.rostro.nombre == "DESCONOCIDO":
            new_vector = self.procesar_frame(frame)
            for name, vector in self.choferes_dict.items():
                result = 1 - spatial.distance.cosine(vector, new_vector)
                if result >= self.confidence_threshold:
                    self.rostro.nombre = name
                    break

class Detector_de_rasgos_faciales(Motor_de_inferencia):

    def suavizar_curva(self, curve):
        for index, point in enumerate(curve):
            if index % 2 and index < len(curve) - 1:
                x = curve[index - 1][1] + (curve[index + 1][1] - curve[index - 1][1]) / 2
                y = curve[index - 1][0] + (curve[index + 1][0] - curve[index - 1][0]) / 2
                point[1] = x
                point[0] = y
        return curve

    def procesar_frame(self, face_frame, rostro):
        drf_result = super().procesar_frame(face_frame)[0]
        self.rostro = rostro
        location = rostro.location
        face_width = location["br"][0] - location["tl"][0]
        face_height = location["br"][1] - location["tl"][1]
        position_points = []
        rows, colums = drf_result[0].shape
        for point in drf_result:
            max_value_index = np.unravel_index(np.argmax(point, axis=None), point.shape)
            position_points.append([
                location["tl"][1] + max_value_index[0] * face_height / rows,
                location["tl"][0] + max_value_index[1] * face_width / colums
                ])
        return position_points

    def detectar_rasgos(self, face_frame):
        position_points = self.procesar_frame(face_frame, Rostro.getInstance())
        self.rostro.margen_rostro = self.suavizar_curva(position_points[:32])
        self.rostro.cejas = position_points[33:51]
        self.rostro.nariz = position_points[52:60]
        self.rostro.ojo_derecho = position_points[61:68] + [position_points[-2]]
        self.rostro.ojo_izquierdo = position_points[69:76] + [position_points[-1]]
        self.rostro.boca = position_points[77:-2]

class Detector_posicion_cabeza(Motor_de_inferencia):
    def conductor_distraido(self):
        if abs(self.rostro.yaw_angle) > 30 or abs(self.rostro.pitch_angle) > 30:
            return True
        return False

    def distraccion_critica(self):
        ret = False
        if self.conductor_distraido():
            self.rostro.contador_distracciones += 1
            if self.rostro.contador_distracciones >= self.rostro.umbral_de_distraccion_critico:
                ret = True
        else:
            self.rostro.contador_distracciones = 0
        return ret

    def get_output_blob_prop(self) -> list:
        return list(self.execution_net.outputs.keys())

    def procesar_frame(self, frame):
        self.rostro.pitch_angle, self.rostro.roll_angle, self.rostro.yaw_angle = super().procesar_frame(frame)

    def detectar_angulos_de_posicion(self, frame):
        self.rostro = Rostro.getInstance()
        self.procesar_frame(frame)
        self.rostro.distraccion_critica = self.distraccion_critica()