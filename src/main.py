import cv2
import motores_de_inferencia as mi
import logging
from objetos import Frame, Imagen, Rostro, COLORS
from imutils import resize
from json import load
from multiprocessing import Process, Value
from time import sleep

def camara_frontal(
        configs,
        distracted,
        accion,
        programa_finalizado):

    device = configs["device"]
    confidence_threshold = configs["confidence_threshold"]
    log = logging.getLogger("Camara Frontal")

    # Creamos los motores de inferencia
    detector_de_rostros = mi.Detector_de_rostros(
        configs["dr_model_xml"],
        configs["dr_model_bin"],
        device,
        confidence_threshold
    )
    identificador_de_rostros = mi.Identificador_de_rostros(
        configs["ir_model_xml"],
        configs["ir_model_bin"],
        device,
        confidence_threshold
    )
    detector_de_rasgos_faciales = mi.Detector_rasgos_faciales(
        configs["drf_model"]
    )
    detector_posicion_cabeza = mi.Detector_posicion_cabeza(
        configs["dpc_model_xml"],
        configs["dpc_model_bin"],
        device,
        confidence_threshold,
        configs["head_grades_threshold"]
    )
    
    #Determinamos la visibilidad de las detecciones
    show_face = configs["show_face"]
    show_name = configs["show_name"]
    show_rasgos_faciales = configs["show_rasgos_faciales"]
    show_posicion_cabeza = configs["show_posicion_cabeza"]
    show_acciones = configs["show_acciones"]
    show_fps = configs["show_fps"]
    
    identificador_de_rostros.generar_base_de_datos_de_choferes(configs["drivers_photos"])
    frame = Frame(configs["front_video_input"])
    success, img = frame.new_frame()
    rostro = Rostro.getInstance()

    acciones = configs["acciones"]

    while success:
        input_height, input_width, _ = img.shape
        detector_de_rostros.procesar_frame(img)
        if rostro.rostro_detectado:
            imagen_rostro_recortado = Imagen.obtener_imagen_rostro_recortado(img)
            identificador_de_rostros.obtener_nombre_conductor(imagen_rostro_recortado)
            alerta_somnolencia, somnolencia_critica = detector_de_rasgos_faciales.detectar_rasgos(imagen_rostro_recortado)
            detector_posicion_cabeza.detectar_angulos_de_posicion(imagen_rostro_recortado)
            distracted.value = rostro.distraccion_critica
            color = COLORS.RED.value if (rostro.distraccion_critica or somnolencia_critica) else COLORS.GREEN.value
            if show_face:
                cv2.rectangle(
                    img, 
                    rostro.location["tl"], 
                    rostro.location["br"], 
                    color, 
                    configs["ancho_recta"]
                )
            if show_posicion_cabeza:
                puntos_de_rotacion = rostro.obtener_puntos_rotacion(input_height, input_width, rostro.location)
                Imagen.dibujar_posicion_cabeza(puntos_de_rotacion, rostro.center, img)
            if rostro.distraccion_critica:
                cv2.putText(
                    img,
                    "DISTRACCION CRITICA",
                    (int(input_width / 4), int(input_height / 15) * 1),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )
            if show_name:
                cv2.putText(
                    img,
                    f"NOMBRE: {rostro.nombre.upper()}",
                    (10, int(input_height / 15)*11),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
            if show_acciones:
                cv2.putText(
                    img,
                    f"ACCION: {acciones[accion.value]}",
                    (10, int(input_height / 15)*12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
            if show_rasgos_faciales:
                Imagen.dibujar_puntos(rostro.obtener_posicion_rasgos_faciales(), color, img)
                detector_de_rasgos_faciales.dibujar_medidor_de_somnolencia(img)
                cv2.putText(
                    img,
                    alerta_somnolencia,
                    (10, int(input_height / 15)*13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
            if show_fps:
                cv2.putText(
                    img,
                    f"FPS: {frame.fps}",
                    (10, int(input_height / 15)*14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )


        showImg = resize(img, height=750, width=680)
        cv2.imshow("showFrontal", showImg)
        cv2.waitKey(1)
        if cv2.waitKey(10) == 27:  # exit if Esc
            programa_finalizado.value = True
            break
        if cv2.waitKey(10) == 97:
            show_face = not show_face
        if cv2.waitKey(10) == 115:
            show_name = not show_name
        if cv2.waitKey(10) == 100:
            show_facial_landmarks = not show_facial_landmarks   

        success, img = frame.new_frame()
    frame.release()
    programa_finalizado.value = True
    log.info("Proceso terminado")

def camara_lateral(configs,
        distracted,
        accion,
        programa_finalizado):
    log = logging.getLogger("Camara lateral")
    show_fps = configs["show_fps"]
    color = COLORS.GREEN.value
    while not programa_finalizado.value:
        frame = Frame(configs["side_video_input"])
        detector_de_acciones_encoder = mi.detector_acciones_encoder(
        configs["dar_model_xml_enc"],
        configs["dar_model_bin_enc"],
        configs["device"],
        configs["confidence_threshold"]
        )
        detector_de_acciones_decoder = mi.detector_acciones_decoder(
            configs["dar_model_xml_dec"],
            configs["dar_model_bin_dec"],
            configs["device"],
            configs["confidence_threshold"]
        )
        while not distracted.value:
            sleep(1)
            if programa_finalizado.value:
                break
        success, img = frame.new_frame()
        while distracted.value and success:
            input_height, _, _ = img.shape
            detector_de_acciones_encoder.procesar_frame(img)
            action_index = detector_de_acciones_decoder.procesar_frame(detector_de_acciones_encoder.frame_queue)
            accion.value = action_index
            if show_fps:
                cv2.putText(
                    img,
                    f"FPS: {frame.fps}",
                    (10, input_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
            cv2.namedWindow('showSide')
            cv2.moveWindow('showSide', 1000, 250)
            showImg = resize(img, height=750, width=680)
            cv2.imshow("showSide", showImg)
            cv2.waitKey(1)
            if programa_finalizado.value:
                break
            success, img = frame.new_frame()
        # Deja de estar distraido, vuelve a conduccion segura
        accion.value = 0
        try:
            if cv2.getWindowProperty('showSide', cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow("showSide")
        except cv2.error as e:
            log.error("ventana showSide cerrada")
        frame.release()
    log.info("Proceso terminado")



def main():

    configs = {}
    # Cargamos configuraciones de json
    with open("config.json") as configs_file:
        configs = load(configs_file)
    distracted = Value('b', False)
    accion = Value('i', 0)
    programa_finalizado = Value('b', False)

    # Proceso que maneja la camara frontal
    frontal_camera_process = Process(target=camara_frontal, args=(
        configs,
        distracted,
        accion,
        programa_finalizado))

    # Proceso que maneja la camara lateral
    side_camera_process = Process(target=camara_lateral, args=(
        configs,
        distracted,
        accion,
        programa_finalizado))

    frontal_camera_process.start()
    side_camera_process.start()

    frontal_camera_process.join()
    side_camera_process.join()

if __name__ == "__main__":
    main()
