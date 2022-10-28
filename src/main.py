import cv2
from imutils import resize
from motor_de_inferencia import Detector_de_rostros

pColor = (0, 0, 255)  # bounding-rect color
rectThinkness = 2

model_bin = "../modelos/face-detection-retail-0004/face-detection-retail-0004.bin"
model_xml = "../modelos/face-detection-retail-0004/face-detection-retail-0004.xml"
#VIDEO_PATH = "./videos_de_ejemplo/video.mp4"
VIDEO_PATH = 0
device = "CPU"
confidence_threshold = 0.8

def main():

    detector_de_rostros = Detector_de_rostros(model_xml, model_bin, device, confidence_threshold)
    vidcap = cv2.VideoCapture(VIDEO_PATH)
    success, img = vidcap.read()
    while success:
        rostro = detector_de_rostros.procesar_frame(img)
        if rostro:
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