import cv2
import math
import numpy as np

from time import time
from enum import Enum

class Frame:
    def __init__(self, video_input: str):
        self.video_cap: tuple = cv2.VideoCapture(video_input)
        self.fps: int = 0
        self.fps_timestamp: int = 0
        self.counter: int = 0

    def new_frame(self) -> tuple:
        self.update_fps()
        return self.video_cap.read()
    def update_fps(self) -> None:
        curr_timestamp = int(time())
        if curr_timestamp > self.fps_timestamp:
            self.fps_timestamp = curr_timestamp
            self.fps = self.counter
            self.counter = 0
        else:
            self.counter += 1
class COLORS(Enum):
    BLUE  = (255,0,0)
    GREEN = (0,255,0)
    RED   = (0,0,255)

    @staticmethod
    def list_values() -> list:
        return list(COLORS._value2member_map_.keys())
class Imagen:

    def __init__(self, batch_size : int, number_of_channels : int, height : int, width : int):
        self.batch_size = batch_size
        self.number_of_channels = number_of_channels
        self.height = height
        self.width = width

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
    @staticmethod
    def obtener_imagen_rostro_recortado(frame):
        rostro = Rostro.getInstance()
        xmin, ymin = rostro.location["tl"]
        xmax, ymax = rostro.location["br"]
        image = frame[ymin : ymax + 1, xmin : xmax + 1]
        if image.any():
            return cv2.resize(
                image,
                (
                    xmax - xmin,
                    ymax - ymin,
                ),
            )
        else:
            return frame
    
    @staticmethod
    def dibujar_posicion_cabeza(points, center, frame):
        colores = COLORS.list_values()
        for index, point in enumerate(points):
            cv2.line(
                frame, (int(center[0]), int(center[1])), point, colores[index], 2
            )
        cv2.circle(frame, points[2], 3, COLORS.BLUE.value, 2)

class Rostro:

    __shared_instance = None

    @staticmethod
    def getInstance():
        """Static Access Method"""
        if not Rostro.__shared_instance:
            Rostro()
        return Rostro.__shared_instance

    def __init__(self):
        if Rostro.__shared_instance:
            raise Exception("This class is a singleton class !")
        else:
            self.nombre = "DESCONOCIDO"
            self.margen_rostro = []
            self.cejas = []
            self. nariz = []
            self.ojo_izquierdo = []
            self. ojo_derecho = []
            self.boca = []
            self.pitch_angle = 0
            self.yaw_angle = 0
            self.roll_angle = 0
            self.contador_distracciones = 0
            self.umbral_de_distraccion_critico = 5
            self.distraccion_critica = False
            self.center = []
            Rostro.__shared_instance = self
    
    def actualizar_atributos(self, inference_result):
        print(f"RESULTADO: {inference_result}")
        self.id = inference_result[0]
        self.label = int(inference_result[1])
        self.confidence = inference_result[2]
        self.location = {
            "tl": [inference_result[3],inference_result[4]],
            "br": [inference_result[5], inference_result[6]]
            }
        self.rostro_detectado = True
    
    def redimensionar_posicion(self, frame_width, frame_height):
        self.location = Imagen.incrementar_area_por_porcentaje(self.location, 10)
        self.location = Imagen.obtener_posicion_en_imagen_original(self.location, frame_width, frame_height)
    
    def obtener_posicion_rasgos_faciales(self):
        return self.margen_rostro + self.nariz + self.ojo_izquierdo + self.ojo_derecho + self.boca

    def obtener_puntos_rotacion(self, height, width, face):
        # Head Pose
        self.center = np.zeros((3, 1), dtype=np.float32)
        self.center[0] = face["tl"][0] + (face["br"][0] - face["tl"][0]) / 2  # x
        self.center[1] = face["tl"][1] + (face["br"][1] - face["tl"][1]) / 2  # y

        ### Draw euler angles 3D axis ###
        pitch = float(self.pitch_angle) * np.float64(np.pi / 180.0)
        yaw = float(self.yaw_angle) * np.float64(np.pi / 180.0)
        roll = float(self.roll_angle) * np.float64(np.pi / 180.0)

        # Euler angles of head rotation in 3d space - (pitch, yaw, roll)
        rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(pitch), -math.sin(pitch)],
                [0, math.sin(pitch), math.cos(pitch)],
            ],
            dtype="double",
        )

        ry = np.array(
            [
                [math.cos(yaw), 0, -math.sin(yaw)],
                [0, 1, 0],
                [math.sin(yaw), 0, math.cos(yaw)],
            ],
            dtype="double",
        )

        rz = np.array(
            [
                [math.cos(roll), -math.sin(roll), 0],
                [math.sin(roll), math.cos(roll), 0],
                [0, 0, 1],
            ],
            dtype="double",
        )

        r = np.dot(rz, np.dot(ry, rx))  # rotation matrix

        # camera intrinsics
        focal_len = 950.0
        cx = width / 2
        cy = height / 2

        camera_mtx = np.zeros((3, 3), dtype=np.float32)
        camera_mtx[0][0] = focal_len
        camera_mtx[0][2] = cx
        camera_mtx[1][1] = focal_len
        camera_mtx[1][2] = cy
        camera_mtx[2][2] = 1

        x_axis = np.zeros((3, 1), dtype=np.float32)
        x_axis[0] = 1 * 50

        y_axis = np.zeros((3, 1), dtype=np.float32)
        y_axis[1] = -1 * 50

        z_axis = np.zeros((3, 1), dtype=np.float32)
        z_axis[2] = -1 * 50

        z_axis1 = np.zeros((3, 1), dtype=np.float32)
        z_axis1[2] = 1 * 50

        dt = np.dtype("float32").type(0)  # type(0) == cv::Scalar(0)
        o = np.zeros((3, 1), dtype=dt)
        o[2] = camera_mtx[0][0]

        x_axis = np.dot(r, x_axis) + o
        y_axis = np.dot(r, y_axis) + o
        z_axis = np.dot(r, z_axis) + o
        z_axis1 = np.dot(r, z_axis1) + o

        points = []

        p2x = int((x_axis[0] / x_axis[2] * camera_mtx[0][0]) + self.center[0])
        p2y = int((x_axis[1] / x_axis[2] * camera_mtx[1][1]) + self.center[1])
        points.append((p2x, p2y))

        p2x = int((y_axis[0] / y_axis[2] * camera_mtx[0][0]) + self.center[0])
        p2y = int((y_axis[1] / y_axis[2] * camera_mtx[1][1]) + self.center[1])
        points.append((p2x, p2y))

        p2x = int((z_axis[0] / z_axis[2] * camera_mtx[0][0]) + self.center[0])
        p2y = int((z_axis[1] / z_axis[2] * camera_mtx[1][1]) + self.center[1])
        points.append((p2x, p2y))

        return points

        