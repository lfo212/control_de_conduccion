import cv2
from imutils import resize
import motores_de_inferencia as mi
from objetos import Imagen, Rostro, COLORS


ancho_recta = 2

# Modelo de deteccion de rostros
dr_model_bin = "../modelos/face-detection-retail-0004/face-detection-retail-0004.bin"
dr_model_xml = "../modelos/face-detection-retail-0004/face-detection-retail-0004.xml"

# Modelo de identificacion de rostros
ir_model_bin = "../modelos/face-reidentification-retail-0095/face-reidentification-retail-0095.bin"
ir_model_xml = "../modelos/face-reidentification-retail-0095/face-reidentification-retail-0095.xml"

# Modelo de deteccion de rasgos faciales
drf_model_bin = "../modelos/facial-landmarks-98-detection-0001/facial-landmarks-98-detection-0001.bin"
drf_model_xml = "../modelos/facial-landmarks-98-detection-0001/facial-landmarks-98-detection-0001.xml"

# Modelo de deteccion de posicion cabeza
dpc_model_bin = "../modelos/head-pose-estimation-adas-0001/head-pose-estimation-adas-0001.bin"
dpc_model_xml = "../modelos/head-pose-estimation-adas-0001/head-pose-estimation-adas-0001.xml"


CHOFERES_PATH = "../imagenes_rostros_conductores"
VIDEO_PATH = 0
device = "CPU"
confidence_threshold = 0.6

def main():

    # Creamos los motores de inferencia
    detector_de_rostros = mi.Detector_de_rostros(dr_model_xml, dr_model_bin, device, confidence_threshold)
    identificador_de_rostros = mi.Identificador_de_rostros(ir_model_xml, ir_model_bin, device, confidence_threshold)
    detector_de_rasgos_faciales = mi.Detector_de_rasgos_faciales(drf_model_xml, drf_model_bin, device, confidence_threshold)
    detector_posicion_cabeza = mi.Detector_posicion_cabeza(dpc_model_xml, dpc_model_bin, device, confidence_threshold)
    
    show_face = True
    show_name = True
    show_rasgos_faciales = True
    show_posicion_cabeza = True
    
    identificador_de_rostros.generar_base_de_datos_de_choferes(CHOFERES_PATH)
    vidcap = cv2.VideoCapture(VIDEO_PATH)
    success, img = vidcap.read()
    rostro = Rostro.getInstance()
    while success:
        input_height, input_width, _ = img.shape
        detector_de_rostros.procesar_frame(img)
        if rostro.rostro_detectado:
            imagen_rostro_recortado = Imagen.obtener_imagen_rostro_recortado(img)
            identificador_de_rostros.obtener_nombre_conductor(imagen_rostro_recortado)
            detector_de_rasgos_faciales.detectar_rasgos(imagen_rostro_recortado)
            detector_posicion_cabeza.detectar_angulos_de_posicion(imagen_rostro_recortado)
            color = COLORS.RED.value if rostro.distraccion_critica else COLORS.GREEN.value
            if show_face:
                cv2.rectangle(
                    img, 
                    rostro.location["tl"], 
                    rostro.location["br"], 
                    color, 
                    ancho_recta
                )
            if show_name:
                cv2.putText(
                    img,
                    rostro.nombre.upper(),
                    (int(input_width / 4), int(input_height / 12) * 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )
            if show_rasgos_faciales:
                for point in rostro.obtener_posicion_rasgos_faciales():
                    cv2.circle(
                        img, 
                        (int(point[1]), int(point[0])),
                        1 + int(0.0012 * 64), 
                        color, 
                        -1
                    )
            if show_posicion_cabeza:
                puntos_de_rotacion = rostro.obtener_puntos_rotacion(input_height, input_width, rostro.location)
                Imagen.dibujar_posicion_cabeza(puntos_de_rotacion, rostro.center, img)
            if rostro.distraccion_critica:
                cv2.putText(
                    img,
                    "DISTRACCION CRITICA",
                    (int(input_width / 4), int(input_height / 12) * 11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    COLORS.RED.value,
                    2,
                )


        showImg = resize(img, height=750, width=680)
        cv2.imshow("showImg", showImg)
        cv2.waitKey(1)
        """
        if cv2.waitKey(10) == 27:  # exit if Esc
            break
        if cv2.waitKey(10) == 97:
            show_face = not show_face
        if cv2.waitKey(10) == 115:
            show_name = not show_name
        if cv2.waitKey(10) == 100:
            show_facial_landmarks = not show_facial_landmarks   
        """

        success, img = vidcap.read()
    print("Programa terminado")

if __name__ == "__main__":
    main()