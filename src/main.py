import cv2
from imutils import resize
import motores_de_inferencia as mi
from objetos import Imagen, Rostro

RED = (0, 0, 255)  # bounding-rect color
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


CHOFERES_PATH = "../imagenes_rostros_conductores"
VIDEO_PATH = 0
device = "CPU"
confidence_threshold = 0.6

def main():

    # Creamos los motores de inferencia
    detector_de_rostros = mi.Detector_de_rostros(dr_model_xml, dr_model_bin, device, confidence_threshold)
    identificador_de_rostros = mi.Identificador_de_rostros(ir_model_xml, ir_model_bin, device, confidence_threshold)
    detector_de_rasgos_faciales = mi.Detector_de_rasgos_faciales(drf_model_xml, drf_model_bin, device, confidence_threshold)
    
    identificador_de_rostros.generar_base_de_datos_de_choferes(CHOFERES_PATH)
    vidcap = cv2.VideoCapture(VIDEO_PATH)
    success, img = vidcap.read()
    while success:
        rostro : Rostro = detector_de_rostros.procesar_frame(img)
        if rostro.rostro_detectado:
            imagen_rostro_recortado = Imagen.obtener_imagen_rostro_recortado(img, rostro)
            if rostro.nombre == "DESCONOCIDO":
                rostro.nombre = identificador_de_rostros.obtener_nombre_conductor(imagen_rostro_recortado)


            drf_results = detector_de_rasgos_faciales.procesar_frame(imagen_rostro_recortado, img.shape, rostro.location["tl"])
            cv2.rectangle(img, rostro.location["tl"], rostro.location["br"], RED, ancho_recta)
            input_height, input_width, _ = img.shape
            cv2.putText(
                img,
                rostro.nombre,
                (int(input_width / 4), int(input_height / 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                RED,
                2,
            )
            for point in drf_results:
                cv2.circle(img, (int(point[1]), int(point[0])), 1 + int(0.0012 * 64), (255, 0, 0), -1)
        showImg = resize(img, height=750, width=680)
        cv2.imshow("showImg", showImg)
        cv2.waitKey(1)
        if cv2.waitKey(10) == 27:  # exit if Esc
            break
        success, img = vidcap.read()
    print("Programa terminado")

if __name__ == "__main__":
    main()