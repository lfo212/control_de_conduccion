import cv2
import motores_de_inferencia as mi
from objetos import Frame, Imagen, Rostro, COLORS
from imutils import resize
from json import load

def main():

    configs = {}
    # Cargamod configuraciones de json
    with open("config.json") as configs_file:
        configs = load(configs_file)
    device = configs["device"]
    confidence_threshold = configs["confidence_threshold"]

    # Creamos los motores de inferencia
    detector_de_rostros = mi.Detector_de_rostros(configs["dr_model_xml"], configs["dr_model_bin"], device, confidence_threshold)
    
    #Determinamos la visibilidad de las detecciones
    show_face = configs["show_face"]
    show_fps = configs["show_fps"]
    frame = Frame(configs["video_input"])
    success, img = frame.new_frame()
    success = True
    rostro = Rostro.getInstance()
    while success:
        input_height, input_width, _ = img.shape
        detector_de_rostros.procesar_frame(img)
        color = COLORS.GREEN.value
        print("CONFIANZA: ", rostro.confidence, end="\r")
        if show_face:
            cv2.rectangle(
                img, 
                rostro.location["tl"], 
                rostro.location["br"], 
                color, 
                configs["ancho_recta"]
            )
        if show_fps:
            cv2.putText(
                img,
                f"FPS: {frame.fps}",
                (10, input_height - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                COLORS.GREEN.value,
                2,
            )

        showImg = resize(img, height=750, width=680)
        cv2.imshow("showImg", showImg)
        key = cv2.waitKey(1)
        if key == 27:  # exit if Esc
            break
        if key == 97:
            show_face = not show_face
        if key == 115:
            show_name = not show_name
        if key == 100:
            show_facial_landmarks = not show_facial_landmarks   

        success, img = frame.new_frame()
    print("Programa terminado")

if __name__ == "__main__":
    main()
