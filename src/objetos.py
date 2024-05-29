import cv2
import math
import numpy as np
import ffmpeg

from time import time
from enum import Enum
from collections import deque

class Frame:
    def __init__(self, video_input: str):
        self.video_cap: tuple = cv2.VideoCapture(video_input)
        self.fps: int = 0
        self.fps_timestamp: int = 0
        self.counter: int = 0
        self.success = False
        self.img = None
        self.frame_queue = deque(maxlen=150) # cola con tamaño maximo de 150 frames, equivalente a 5 seg a 30 fps

    def new_frame(self) -> tuple:
        self.update_fps()
        self.success, self.img = self.video_cap.read()

    def add_frame_to_queue(self):
        self.frame_queue.append(self.img)

    def update_fps(self) -> None:
        curr_timestamp = int(time())
        if curr_timestamp > self.fps_timestamp:
            self.fps_timestamp = curr_timestamp
            self.fps = self.counter
            self.counter = 0
        else:
            self.counter += 1

    def release(self):
        self.video_cap.release()
class COLORS(Enum):
    BLUE  = (255,0,0)
    GREEN = (0,255,0)
    RED   = (0,0,255)

    @staticmethod
    def list_values() -> list:
        return list(COLORS._value2member_map_.keys())
class Imagen:

    # Drowsiness levels
    MAX_NORMAL = 20
    MAX_WARNING = 50
    MAX_CRITICAL = 70
    MAX_CRITICAL_VISUAL = 100

    def __init__(self, parameters):
        if len(parameters) == 4:
            self.batch_size, self.number_of_channels, self.height, self.width = parameters
            self.secuence = False
        else:
            self.batch_size, self.duration, self.dimension_embedding = parameters
            self.secuence = True

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
                frame, (int(center["x"]), int(center["y"])), point, colores[index], 2
            )
        cv2.circle(frame, points[2], 3, COLORS.BLUE.value, 2)
    
    @staticmethod
    def dibujar_puntos(points, color, frame):
        for point in points:
            cv2.circle(
                frame, 
                (int(point[0]), int(point[1])),
                1 + int(0.0012 * 64), 
                color,
                -1
            )
    
    @staticmethod
    def dibujar_medidor_de_somnolencia(somnolencia, frame, log):
        _ , width, _ = frame.shape
        x = 200 - 125
        y = 125
        x_vum = 20
        y_vum = 150
        y_vum_unit = 1.5
        x_truck_i = width - (x + 10)
        y_driver_i = y + 30
        y_driver = y - 60
        y_alarm = y_driver_i + y_driver + 10
        x_vum_draw = x_truck_i + 15
        y_vum_draw = y_alarm + 55
        line_width = -1
        if somnolencia <= Imagen.MAX_NORMAL:
            frame = cv2.rectangle(
                frame,
                (x_vum_draw, y_vum_draw + y_vum - int(y_vum_unit * somnolencia)),
                (x_vum_draw + x_vum, y_vum_draw + y_vum),
                (0, 255, 0),
                line_width,
            )
            log.debug(f"Drowsiness level is normal: {somnolencia}")
        elif Imagen.MAX_NORMAL < somnolencia <= Imagen.MAX_WARNING:
            frame = cv2.rectangle(
                frame,
                (x_vum_draw, y_vum_draw + y_vum - int(y_vum_unit * Imagen.MAX_NORMAL)),
                (x_vum_draw + x_vum, y_vum_draw + y_vum),
                (0, 255, 0),
                line_width,
            )
            frame = cv2.rectangle(
                frame,
                (x_vum_draw, y_vum_draw + y_vum - int(y_vum_unit * somnolencia)),
                (x_vum_draw + x_vum, y_vum_draw + y_vum - int(y_vum_unit * Imagen.MAX_NORMAL)),
                (0, 255, 255),
                line_width,
            )
            log.debug(
                f"Drowsiness level higher than normal but less than warning: {somnolencia}"
            )
        elif Imagen.MAX_WARNING < somnolencia <= Imagen.MAX_CRITICAL:
            frame = cv2.rectangle(
                frame,
                (x_vum_draw, y_vum_draw + y_vum - int(y_vum_unit * Imagen.MAX_NORMAL)),
                (x_vum_draw + x_vum, y_vum_draw + y_vum),
                (0, 255, 0),
                line_width,
            )
            frame = cv2.rectangle(
                frame,
                (x_vum_draw, y_vum_draw + y_vum - int(y_vum_unit * Imagen.MAX_WARNING)),
                (x_vum_draw + x_vum, y_vum_draw + y_vum - int(y_vum_unit * Imagen.MAX_NORMAL)),
                (0, 255, 255),
                line_width,
            )
            frame = cv2.rectangle(
                frame,
                (x_vum_draw, y_vum_draw + y_vum - int(y_vum_unit * somnolencia)),
                (
                    x_vum_draw + x_vum,
                    y_vum_draw + y_vum - int(y_vum_unit * Imagen.MAX_WARNING),
                ),
                (0, 0, 255),
                line_width,
            )
            log.debug(
                f"Drowsiness level higher than warning but less than critical: {somnolencia}"
            )
        else:
            frame = cv2.rectangle(
                frame,
                (x_vum_draw, y_vum_draw + y_vum - int(y_vum_unit * Imagen.MAX_NORMAL)),
                (x_vum_draw + x_vum, y_vum_draw + y_vum),
                (0, 255, 0),
                line_width,
            )
            frame = cv2.rectangle(
                frame,
                (x_vum_draw, y_vum_draw + y_vum - int(y_vum_unit * Imagen.MAX_WARNING)),
                (x_vum_draw + x_vum, y_vum_draw + y_vum - int(y_vum_unit * Imagen.MAX_NORMAL)),
                (0, 255, 255),
                line_width,
            )
            frame = cv2.rectangle(
                frame,
                (
                    x_vum_draw,
                    y_vum_draw + y_vum - int(y_vum_unit * Imagen.MAX_CRITICAL_VISUAL),
                ),
                (
                    x_vum_draw + x_vum,
                    y_vum_draw + y_vum - int(y_vum_unit * Imagen.MAX_WARNING),
                ),
                (0, 0, 255),
                line_width,
            )
            log.debug(f"Drowsiness level critical: {somnolencia}")

        cv2.putText(
            frame,
            str(int(somnolencia)),
            (
                x_vum_draw + 30,
                y_vum_draw
                + y_vum
                - int(y_vum_unit * min(somnolencia, Imagen.MAX_CRITICAL_VISUAL))
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

class Rostro():

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
        self.id = inference_result[0]
        self.label = int(inference_result[1])
        self.confidence = inference_result[2]
        self.location = {
            "tl": [inference_result[3],inference_result[4]],
            "br": [inference_result[5], inference_result[6]]
            }
        self.rostro_detectado = True
    
    def actualizar_referencia(self, point):
        return [self.location["tl"][0] + point[0], self.location["tl"][1] + point[1]]

    def actualizar_referencia_lista(self, points):
        return [self.actualizar_referencia(point) for point in points]
    
    def redimensionar_posicion(self, frame_width, frame_height):
        self.location = Imagen.incrementar_area_por_porcentaje(self.location, 10)
        self.location = Imagen.obtener_posicion_en_imagen_original(self.location, frame_width, frame_height)
        self.rect = {
            "x": self.location["tl"][0],
            "y": self.location["tl"][1],
            "width": self.location["br"][0] - self.location["tl"][0],
            "height": self.location["br"][1] - self.location["tl"][1],
        }
        self.center = {
            "x": self.rect["x"] + self.rect["width"] / 2,
            "y": self.rect["y"] + self.rect["height"] / 2,
        }
    

    
    def obtener_posicion_rasgos_faciales(self):
        return self.ojo_izquierdo + self.ojo_derecho + self.boca

    def obtener_puntos_rotacion(self, height, width, face):
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

        p2x = int((x_axis[0] / x_axis[2] * camera_mtx[0][0]) + self.center["x"])
        p2y = int((x_axis[1] / x_axis[2] * camera_mtx[1][1]) + self.center["y"])
        points.append((p2x, p2y))

        p2x = int((y_axis[0] / y_axis[2] * camera_mtx[0][0]) + self.center["x"])
        p2y = int((y_axis[1] / y_axis[2] * camera_mtx[1][1]) + self.center["y"])
        points.append((p2x, p2y))

        p2x = int((z_axis[0] / z_axis[2] * camera_mtx[0][0]) + self.center["x"])
        p2y = int((z_axis[1] / z_axis[2] * camera_mtx[1][1]) + self.center["y"])
        points.append((p2x, p2y))

        return points

class Encoder:
    def __init__(
        self,
        filename,
        fps=7,
        input_args={},
        output_args={},
    ):
        self.filename = filename
        self.process = None
        self.input_args = input_args
        self.output_args = output_args
        self.input_args["framerate"] = fps if fps > 0 else 7
        self.input_args["pix_fmt"] = "bgr24"
        self.output_args["pix_fmt"] = "yuv420p"
        self.output_args["vcodec"] = "libx264"

    def add_frame(self, frame):
        if self.process is None:
            h, w = frame.shape[:2]
            self.process = (
                ffmpeg.input(
                    "pipe:",
                    format="rawvideo",
                    s="{}x{}".format(w, h),
                    **self.input_args
                )
                .output(self.filename, **self.output_args)
                .overwrite_output()
                .run_async(pipe_stdin=True)
            )
        self.process.stdin.write(frame.astype(np.uint8).tobytes())

    def close(self):
        if self.process is None:
            return
        self.process.stdin.close()
        self.process.wait()

    def add(self, frame):
        #frame = blob2matrix(frame, width, height)
        self.add_frame(frame)