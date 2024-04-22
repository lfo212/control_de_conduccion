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
    #identificador_de_rostros = mi.Identificador_de_rostros(configs["ir_model_xml"], configs["ir_model_bin"], device, confidence_threshold)
    detector_de_rasgos_faciales = mi.Detector_rasgos_faciales(configs["drf_model"], device)
    #detector_posicion_cabeza = mi.Detector_posicion_cabeza(configs["dpc_model_xml"], configs["dpc_model_bin"], device, confidence_threshold)
    
    #Determinamos la visibilidad de las detecciones
    show_face = configs["show_face"]
    show_name = configs["show_name"]
    show_rasgos_faciales = configs["show_rasgos_faciales"]
    show_posicion_cabeza = configs["show_posicion_cabeza"]
    show_fps = configs["show_fps"]
    
    #identificador_de_rostros.generar_base_de_datos_de_choferes(configs["drivers_photos"])
    frame = Frame(configs["video_input"])
    success, img = frame.new_frame()
    rostro = Rostro.getInstance()
    while success:
        input_height, input_width, _ = img.shape
        detector_de_rostros.procesar_frame(img)
        if rostro.rostro_detectado:
            #imagen_rostro_recortado = Imagen.obtener_imagen_rostro_recortado(img)
            #identificador_de_rostros.obtener_nombre_conductor(imagen_rostro_recortado)
            img = detector_de_rasgos_faciales.detectar_rasgos(img, rostro.location)
            print(detector_de_rasgos_faciales.sleepDetect)
            #detector_posicion_cabeza.detectar_angulos_de_posicion(imagen_rostro_recortado)
            color = COLORS.RED.value if rostro.distraccion_critica else COLORS.GREEN.value
            if show_face:
                cv2.rectangle(
                    img, 
                    rostro.location["tl"], 
                    rostro.location["br"], 
                    color, 
                    configs["ancho_recta"]
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
        cv2.waitKey(1)
        if cv2.waitKey(10) == 27:  # exit if Esc
            break
        if cv2.waitKey(10) == 97:
            show_face = not show_face
        if cv2.waitKey(10) == 115:
            show_name = not show_name
        if cv2.waitKey(10) == 100:
            show_facial_landmarks = not show_facial_landmarks   

        success, img = frame.new_frame()
    print("Programa terminado")

if __name__ == "__main__":
    main()
