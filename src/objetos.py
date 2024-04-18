import cv2

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
            self.margen_rostro = []
            self.center = []
            Rostro.__shared_instance = self
    
    def actualizar_atributos(self, inference_result):
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