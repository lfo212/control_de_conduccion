import cv2
from imutils import resize
from motores_de_inferencia import Detector_de_rostros, Identificador_de_rostros
from objetos import Imagen, Rostro

RED = (0, 0, 255)  # bounding-rect color
ancho_recta = 2

dr_model_bin = "../modelos/face-detection-retail-0004/face-detection-retail-0004.bin"
dr_model_xml = "../modelos/face-detection-retail-0004/face-detection-retail-0004.xml"
ir_model_bin = "../modelos/face-reidentification-retail-0095/face-reidentification-retail-0095.bin"
ir_model_xml = "../modelos/face-reidentification-retail-0095/face-reidentification-retail-0095.xml"
CHOFERES_PATH = "../imagenes_rostros_conductores"
VIDEO_PATH = 0
device = "CPU"
confidence_threshold = 0.8

def main():

    detector_de_rostros = Detector_de_rostros(dr_model_xml, dr_model_bin, device, confidence_threshold)
    identificador_de_rostros = Identificador_de_rostros(ir_model_xml, ir_model_bin, device, confidence_threshold)
    identificador_de_rostros.generar_base_de_datos_de_choferes(CHOFERES_PATH)
    vidcap = cv2.VideoCapture(VIDEO_PATH)
    success, img = vidcap.read()
    while success:
        rostro = detector_de_rostros.procesar_frame(img)
        if rostro.rostro_detectado:
            imagen_rostro_recortado = Imagen.obtener_imagen_rostro_recortado(img, rostro)
            if rostro.nombre == "DESCONOCIDO":
                rostro.nombre = identificador_de_rostros.obtener_nombre_conductor(imagen_rostro_recortado)
            cv2.rectangle(img, rostro.location["tl"], rostro.location["br"], RED, ancho_recta)
            input_height, input_width, _ = img.shape
            cv2.putText(
                img,
                rostro.nombre,
                (int(input_width / 4), int(input_height / 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (255, 255, 255),
                2,
            )

        showImg = resize(img, height=750, width=680)
        cv2.imshow("showImg", showImg)
        cv2.waitKey(1)
        if cv2.waitKey(10) == 27:  # exit if Esc
            break
        success, img = vidcap.read()
    print("Programa terminado")


if __name__ == "__main__":
    main()