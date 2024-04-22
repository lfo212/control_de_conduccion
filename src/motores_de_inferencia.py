import cv2
import os
import logging
import numpy as np
import dlib

from openvino.inference_engine import IECore
from objetos import Imagen, Rostro, COLORS
from imutils import paths, face_utils
from scipy import spatial
from time import time
from math import sqrt

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

"""
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

"""

class Detector_rasgos_faciales():
    
    SLEEP_MESSAGE = {
        0: "OK",
        1: "DROWSINESS WARNING LOW",
        2: "DROWSINESS WARNING HIGH",
        3: "DROWSINESS CRITICAL",
    }

    # Drowsiness levels
    MAX_NORMAL = 20
    MAX_WARNING = 50
    MAX_CRITICAL = 70
    MAX_CRITICAL_VISUAL = 100

    def __init__(self, shape_predictor, device):
        """Constructor"""
        self.log = logging.getLogger("RASGOS_FACIALES")
        self.contador_ojos_cerrados = 0
        self.limite_ojos_cerrados = 10
        self.limite_cierre_ojos = 0.215
        self.limite_apertura_boca = 0.65
        self.contador_bostezos = 0
        self.limite_bostezos = 5
        self.somnolencia = 0
        self.primer_pestaneo = True
        self.tiempo_pestaneo = 0.0
        self.pestaneos_totales = 0
        self.bostezos_totales = 0
        self.start_time = None
        self.v_pestaneo = 0
        self.v_bostezo = 0

        if not os.path.exists(shape_predictor):
            raise FileNotFoundError(f"Shape predictor missing: {shape_predictor}")
        self.log.debug("Config reading completed...")
        self.predictor = dlib.shape_predictor(shape_predictor)
        self.detector = dlib.get_frontal_face_detector()

    def detectar_rasgos(self, rostro_recortado):
        rostro = Rostro.getInstance()
        shape = self.predictor(rostro_recortado, self.convert_to_dlib_rect(rostro_recortado))
        shape = face_utils.shape_to_np(shape)
        shape = rostro.actualizar_referencia_lista(shape)
        rostro.ojo_izquierdo = shape[36:42]
        rostro.ojo_derecho = shape[42:48]
        rostro.boca = shape[48:59][::2]
        self.calcular_ear(rostro)
        self.calcular_mar(rostro)
        nivel_somnolencia = self.calcular_somnolencia()
        self.ret_message = {
            "Pestaneos_totales": self.pestaneos_totales,
            "Mensaje": self.SLEEP_MESSAGE[nivel_somnolencia],
            "somnolencia": self.somnolencia,
            "Bostezos_totales": self.bostezos_totales
        }
        return self.ret_message

    def convert_to_dlib_rect(self, frame):
        height, width, _ = frame.shape
        scale_factor_x = 0.15
        scale_factor_y = 0.20
        x = scale_factor_x * width
        y = scale_factor_y * height
        width = width * (1 -2 * scale_factor_x)
        height = height * (1 - scale_factor_y)
        return dlib.rectangle(
            left=int(x),
            top=int(y),
            right=int(x + width),
            bottom=int(y + height)
        )

    def dist_a_to_b(self, a, b):
        return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def calcular_relacion_de_aspecto(self, object):
        return (
            self.dist_a_to_b(object[1], object[5])
            + self.dist_a_to_b(object[2], object[4])
        ) / (2 * self.dist_a_to_b(object[0], object[3]))

    def calcular_ear(self, rostro):
        # Calculo de relacion de aspecto del ojo (EAR)
        ear_izq = self.calcular_relacion_de_aspecto(rostro.ojo_izquierdo)
        ear_der = self.calcular_relacion_de_aspecto(rostro.ojo_derecho)
        ear_avg = (ear_izq + ear_der) / 2

        if ear_avg < self.limite_cierre_ojos:
            self.v_pestaneo = 1
            if self.primer_pestaneo:
                self.log.debug(f"PRIMER PESTANEO. contador: {self.contador_ojos_cerrados}")
                self.tiempo_pestaneo = 0.0
                self.primer_pestaneo = False
            else:
                self.tiempo_pestaneo = (time() - self.start_time) * 1000 - self.tiempo_pestaneo
                self.log.debug(f"tiempo de pestaneo: {self.tiempo_pestaneo}")
            self.start_time = time()
            self.contador_ojos_cerrados += 1
            self.log.debug(f"contador ojos cerrados: {self.contador_ojos_cerrados}")

        else:
            if self.contador_ojos_cerrados >= self.limite_ojos_cerrados:
                self.pestaneos_totales += 1
            self.contador_ojos_cerrados = 0
            self.v_pestaneo = 0
            self.primer_pestaneo = True
            self.tiempo_pestaneo = 0.0

    def calcular_mar(self, rostro):
        # Calculo de relacion de aspecto de la boca
        boca_ar = self.calcular_relacion_de_aspecto(rostro.boca)
        if boca_ar > self.limite_apertura_boca:
            self.contador_bostezos += 1
        else:
            if self.contador_bostezos >= self.limite_bostezos:
                self.bostezos_totales += 1
                self.v_bostezo = 1
            self.contador_bostezos = 0


    def calcular_nivel_somnolencia(self, somnolencia):
        asignaciones = {
            (0, self.MAX_NORMAL): 0,
            (self.MAX_NORMAL, self.MAX_WARNING): 1,
            (self.MAX_WARNING, self.MAX_CRITICAL_VISUAL): 2,
            (self.MAX_CRITICAL_VISUAL, float('inf')): 3
        }
        for rango, valor in asignaciones.items():
            if rango[0] <= somnolencia < rango[1]:
                return valor

    def calcular_somnolencia(self):
        self.v_bostezo = 0
        if self.somnolencia <= self.MAX_NORMAL and (self.v_bostezo != 0 or self.v_pestaneo != 0):
            self.somnolencia = int(
                self.somnolencia
                + 100 * self.v_bostezo
                + 50 * self.v_pestaneo * (self.tiempo_pestaneo / 1000)
            )
        elif self.somnolencia > self.MAX_NORMAL and (self.v_bostezo != 0 or self.v_pestaneo != 0):
            self.somnolencia = int(
                self.somnolencia
                + 50 * self.v_bostezo
                + 100 * self.v_pestaneo * (self.tiempo_pestaneo / 1000)
            )
        else:
            if self.somnolencia >= 1:
                self.somnolencia -= 8
            if self.somnolencia < 0:
                self.somnolencia = 0
            self.log.debug(f"Disminuye somnolencia {self.somnolencia}")
        
        

        return self.calcular_nivel_somnolencia(self.somnolencia)

    def dibujar_medidor_de_somnolencia(self, frame):
        self.height, self.width, _ = frame.shape
        x = 200 - 125
        y = 125
        x_vum = 20
        y_vum = 150
        y_vum_unit = 1.5
        x_truck_i = self.width - (x + 10)
        y_driver_i = y + 30
        y_driver = y - 60
        y_alarm = y_driver_i + y_driver + 10
        x_vum_draw = x_truck_i + 15
        y_vum_draw = y_alarm + 55
        line_width = -1
        if self.somnolencia <= self.MAX_NORMAL:
            frame = cv2.rectangle(
                frame,
                (x_vum_draw, y_vum_draw + y_vum - int(y_vum_unit * self.somnolencia)),
                (x_vum_draw + x_vum, y_vum_draw + y_vum),
                (0, 255, 0),
                line_width,
            )
            self.log.debug(f"Drowsiness level is normal: {self.somnolencia}")
        elif self.MAX_NORMAL < self.somnolencia <= self.MAX_WARNING:
            frame = cv2.rectangle(
                frame,
                (x_vum_draw, y_vum_draw + y_vum - int(y_vum_unit * self.MAX_NORMAL)),
                (x_vum_draw + x_vum, y_vum_draw + y_vum),
                (0, 255, 0),
                line_width,
            )
            frame = cv2.rectangle(
                frame,
                (x_vum_draw, y_vum_draw + y_vum - int(y_vum_unit * self.somnolencia)),
                (x_vum_draw + x_vum, y_vum_draw + y_vum - int(y_vum_unit * self.MAX_NORMAL)),
                (0, 255, 255),
                line_width,
            )
            self.log.debug(
                f"Drowsiness level higher than normal but less than warning: {self.somnolencia}"
            )
        elif self.MAX_WARNING < self.somnolencia <= self.MAX_CRITICAL:
            frame = cv2.rectangle(
                frame,
                (x_vum_draw, y_vum_draw + y_vum - int(y_vum_unit * self.MAX_NORMAL)),
                (x_vum_draw + x_vum, y_vum_draw + y_vum),
                (0, 255, 0),
                line_width,
            )
            frame = cv2.rectangle(
                frame,
                (x_vum_draw, y_vum_draw + y_vum - int(y_vum_unit * self.MAX_WARNING)),
                (x_vum_draw + x_vum, y_vum_draw + y_vum - int(y_vum_unit * self.MAX_NORMAL)),
                (0, 255, 255),
                line_width,
            )
            frame = cv2.rectangle(
                frame,
                (x_vum_draw, y_vum_draw + y_vum - int(y_vum_unit * self.somnolencia)),
                (
                    x_vum_draw + x_vum,
                    y_vum_draw + y_vum - int(y_vum_unit * self.MAX_WARNING),
                ),
                (0, 0, 255),
                line_width,
            )
            self.log.debug(
                f"Drowsiness level higher than warning but less than critical: {self.somnolencia}"
            )
        else:
            frame = cv2.rectangle(
                frame,
                (x_vum_draw, y_vum_draw + y_vum - int(y_vum_unit * self.MAX_NORMAL)),
                (x_vum_draw + x_vum, y_vum_draw + y_vum),
                (0, 255, 0),
                line_width,
            )
            frame = cv2.rectangle(
                frame,
                (x_vum_draw, y_vum_draw + y_vum - int(y_vum_unit * self.MAX_WARNING)),
                (x_vum_draw + x_vum, y_vum_draw + y_vum - int(y_vum_unit * self.MAX_NORMAL)),
                (0, 255, 255),
                line_width,
            )
            frame = cv2.rectangle(
                frame,
                (
                    x_vum_draw,
                    y_vum_draw + y_vum - int(y_vum_unit * self.MAX_CRITICAL_VISUAL),
                ),
                (
                    x_vum_draw + x_vum,
                    y_vum_draw + y_vum - int(y_vum_unit * self.MAX_WARNING),
                ),
                (0, 0, 255),
                line_width,
            )
            self.log.debug(f"Drowsiness level critical: {self.somnolencia}")

        cv2.putText(
            frame,
            str(int(self.somnolencia)),
            (
                x_vum_draw + 30,
                y_vum_draw
                + y_vum
                - int(y_vum_unit * min(self.somnolencia, self.MAX_CRITICAL_VISUAL))
                + 5,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
        )

        cv2.rectangle(
            frame,
            (x_vum_draw, y_vum_draw),
            (x_vum_draw + x_vum, y_vum_draw + y_vum),
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            "Medidor de",
            (x_vum_draw - 35, y_vum_draw - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            "somnolencia",
            (x_vum_draw - 35, y_vum_draw - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        return frame

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