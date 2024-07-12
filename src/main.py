import cv2
import motores_de_inferencia as mi
import logging
import asyncio
import websockets
import base64
import os
from datetime import datetime
from objetos import Frame, Imagen, Rostro, COLORS, Encoder
from json import load
from multiprocessing import Process, Value, Manager
import concurrent.futures

def create_video_clip(
    frame,
    accion,
    conductor,
    log
):
    utc_time = datetime.now()
    utc_time.strftime("%Y-%m-%d-%H-%M-%S")
    filename = (conductor+"_"+accion+"_"+str(utc_time)[:-7]).replace(" ","_")
    file_path = os.path.join(
        "eventos", filename+ ".mp4"
    )
    while not os.path.isfile(file_path):
        video = Encoder(filename=file_path, fps=frame.fps)
        for image in list(frame.frame_queue):
            try:
                video.add(image)
            except Exception as e:
                log.error(f"Error al crear el video: {e}")
                video.close()
                return            
        video.close()
    log.info(f"Video record finished: {filename}")

def graficar_resultados(frame, configs, rostro, accion, log):
    input_height, input_width, _ = frame.img.shape
    color = COLORS.RED.value if (
        rostro.distraccion_critica or
        rostro.somnolencia_critica or
        bool(accion.value) or
        not rostro.habilitado) else COLORS.GREEN.value
    accion_string = configs["acciones"][accion.value]
    if configs["show_face"]:
        cv2.rectangle(
            frame.img,
            rostro.location["tl"],
            rostro.location["br"],
            color,
            configs["ancho_recta"]
        )
    if configs["show_posicion_cabeza"]:
        puntos_de_rotacion = rostro.obtener_puntos_rotacion(input_height, input_width, rostro.location)
        Imagen.dibujar_posicion_cabeza(puntos_de_rotacion, rostro.center, frame.img)
    if rostro.distraccion_critica or bool(accion.value):
        cv2.putText(
            frame.img,
            "DISTRACCION CRITICA",
            (int(input_width / 4), int(input_height / 15) * 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
    if configs["show_name"]:
        cv2.putText(
            frame.img,
            f"NOMBRE: {rostro.nombre.upper()} {'(conduccion habilitada)' if rostro.habilitado else '(no habilitado)'}",
            (10, int(input_height / 15)*11),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    if configs["show_acciones"]:
        cv2.putText(
            frame.img,
            f"ACCION: {accion_string}",
            (10, int(input_height / 15)*12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    if configs["show_rasgos_faciales"]:
        Imagen.dibujar_puntos(rostro.obtener_posicion_rasgos_faciales(), color, frame.img)
        Imagen.dibujar_medidor_de_somnolencia(rostro.somnolencia, frame.img, log)
        cv2.putText(
            frame.img,
            rostro.alerta_somnolencia,
            (10, int(input_height / 15)*13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )
    if configs["show_fps"]:
        cv2.putText(
            frame.img,
            f"FPS: {frame.fps}",
            (10, int(input_height / 15)*14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

def camara_frontal(
        configs,
        distracted,
        accion,
        conductor,
        programa_finalizado,
        frame_frontal):

    log = logging.getLogger("Camara Frontal")
    device = configs["device"]
    confidence_threshold = float(configs["confidence_threshold"])
    rostro = Rostro.getInstance()
    frame = Frame(configs["front_video_input"])

    #Creamos los motores de inferencia
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
        float(configs["head_grades_threshold"])
    )

    # Generamos base de datos de choferes
    identificador_de_rostros.generar_base_de_datos_de_choferes(configs["drivers_photos"], detector_de_rostros)


    # Thread pool executor for running tasks concurrently
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    while True:
        frame.new_frame()
        if frame.success:
            detector_de_rostros.procesar_frame(frame.img)
            if rostro.rostro_detectado:
                frame.imagen_rostro_recortado = Imagen.obtener_imagen_rostro_recortado(frame.img)
                # Ejecuto la inferencia del resto de modelos de forma concurrente
                futures = []
                futures.append(executor.submit(identificador_de_rostros.obtener_nombre_conductor, frame))
                futures.append(executor.submit(detector_de_rasgos_faciales.detectar_rasgos, frame))
                futures.append(executor.submit(detector_posicion_cabeza.detectar_angulos_de_posicion, frame))

                # Espero a que todas las tareas se completen
                concurrent.futures.wait(futures)
                # Proceso los resultados
                conductor[0] = rostro.nombre
                rostro.habilitado = rostro.nombre == configs["conductor"]
                distracted.value = rostro.distraccion_critica
                graficar_resultados(frame, configs, rostro, accion, log)
            with frame_frontal["lock"]:
                frame_frontal["img"] = frame.img
            frame.new_frame()
        else:
            try:
                frame.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reinicio en caso de ser un video
            except:
                frame.release()
                programa_finalizado.value = True
                log.info("Proceso terminado")

def camara_lateral(configs,
        distracted,
        accion,
        conductor,
        programa_finalizado,
        frame_lateral):
    log = logging.getLogger("Camara lateral")
    show_fps = configs["show_fps"]
    color = COLORS.GREEN.value
    frame = Frame(configs["side_video_input"])
    acciones = configs["acciones"]
    accion_previa = None
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
    while not programa_finalizado.value:
        frame.new_frame()
        frame.add_frame_to_queue()
        if frame.success:
            input_height, _, _ = frame.img.shape
            if distracted.value:
                detector_de_acciones_encoder.procesar_frame(frame.img)
                action_index = detector_de_acciones_decoder.procesar_secuencia(detector_de_acciones_encoder.frame_queue)
                accion.value = action_index
                if bool(accion.value) and accion.value != accion_previa:
                    accion_previa = accion.value
                    create_video_clip(frame, acciones[accion.value], conductor[0], log)
            else:
                accion.value = 0
            if show_fps:
                cv2.putText(
                    frame.img,
                    f"FPS: {frame.fps}",
                    (10, input_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
            with frame_lateral["lock"]:
                frame_lateral["img"] = frame.img
        else:
            frame.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame.release()
    log.info("Proceso terminado")

async def send_frame(
    websocket,
    path,
    data
):
    log = logging.getLogger("Send frame")
    while True:
        with data["lock"]:
            if data["img"] is not None:
                _, buffer = cv2.imencode('.jpg', data["img"])
                frame_encoded = base64.b64encode(buffer).decode('utf-8')
                try:
                    await websocket.send(frame_encoded)
                except websockets.exceptions.ConnectionClosed:
                    log.error("WebSocket connection closed")
                    break

            await asyncio.sleep(0.033)
    log.info("Proceso terminado")


def main():

    logging.basicConfig(level=logging.INFO)  # Set logging level to INFO

    configs = {}
    # Cargamos configuraciones de json
    with open("config.json") as configs_file:
        configs = load(configs_file)
    #Declaramos variables compartidas
    distracted = Value('b', False)
    accion = Value('i', 0)
    programa_finalizado = Value('b', False)
    manager = Manager()
    conductor = manager.list()
    conductor.append("Desconocido")
    frame_frontal = manager.dict()
    frame_lateral = manager.dict()
    frame_frontal["img"] = None
    frame_frontal["lock"] = manager.Lock()
    frame_lateral["img"] = None
    frame_lateral["lock"] = manager.Lock()

    # Proceso que maneja la camara frontal
    frontal_camera_process = Process(target=camara_frontal, args=(
        configs,
        distracted,
        accion,
        conductor,
        programa_finalizado,
        frame_frontal))
    # Proceso que maneja la camara lateral
    side_camera_process = Process(target=camara_lateral, args=(
        configs,
        distracted,
        accion,
        conductor,
        programa_finalizado,
        frame_lateral))
    # Tarea que maneja el streaming de la camara frontal
    start_server_frontal = websockets.serve(
        lambda ws, path: send_frame(ws, path, frame_frontal),
        "0.0.0.0",
        8765
    )
    # Tarea que maneja el streaming de la camara lateral
    start_server_lateral = websockets.serve(
        lambda ws, path: send_frame(ws, path, frame_lateral),
        "0.0.0.0",
        8766
    )

    frontal_camera_process.start()
    side_camera_process.start()

    # Run the WebSocket server
    asyncio.get_event_loop().run_until_complete(start_server_frontal)
    asyncio.get_event_loop().run_until_complete(start_server_lateral)
    asyncio.get_event_loop().run_forever()

    frontal_camera_process.join()
    side_camera_process.join()

if __name__ == "__main__":
    main()
