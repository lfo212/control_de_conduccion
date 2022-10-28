import cv2
from imutils import resize
from motores_de_inferencia import Detector_de_rostros, Identificador_de_rostros
from objetos import Imagen

pColor = (0, 0, 255)  # bounding-rect color
rectThinkness = 2

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
        if rostro:
            imagen_rostro_recortado = Imagen.obtener_imagen_rostro_recortado(img, rostro)
            nombre = identificador_de_rostros.obtener_nombre_conductor(imagen_rostro_recortado)
            print(nombre)
            cv2.rectangle(img, rostro["tl"], rostro["br"], pColor, rectThinkness)

        showImg = resize(img, height=750, width=680)
        cv2.imshow("showImg", showImg)
        cv2.waitKey(1)
        if cv2.waitKey(10) == 27:  # exit if Esc
            break
        success, img = vidcap.read()
    print("Programa terminado")


if __name__ == "__main__":
    main()