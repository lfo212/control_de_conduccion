import unittest
from unittest.mock import MagicMock
from objetos import Frame, Rostro
from motores_de_inferencia import Detector_de_rostros, Motor_de_inferencia, Identificador_de_rostros, Detector_rasgos_faciales

class TestFrame(unittest.TestCase):
    def setUp(self):
        self.frame = Frame("test_video.mp4")

    def test_update_fps(self):
        # Mocking time() para devolver un valor predefinido
        self.frame.counter = 5
        with unittest.mock.patch('time.time', return_value=10):
            self.frame.update_fps()
        self.assertEqual(self.frame.fps, 5)

class TestDetectorDeRostros(unittest.TestCase):
    def setUp(self):
        self.detector = Detector_de_rostros("../modelos/face-detection-retail-0004/face-detection-retail-0004.xml", "../modelos/face-detection-retail-0004/face-detection-retail-0004.bin", "CPU", 0.5)
        # Mocking super().procesar_frame() para retornar un valor predefinido
        Motor_de_inferencia.procesar_frame = MagicMock(return_value=[[[[0, 1, 2, 3, 4, 5, 6]]]])
        self.frame = Frame(0)
        self.frame.shape = [100, 100, 100]

    def test_procesar_frame(self):
        
        rostro = self.detector.procesar_frame(self.frame)
        # Verificando el tipo de rostro
        self.assertIsInstance(rostro, Rostro)
        # Verificando location
        self.assertEqual(rostro.location, {"tl": (270,360), "br": (100,100)})
        # Verificando confidence
        self.assertEqual(rostro.confidence, 2)

class TestIdentificadorDeRostros(unittest.TestCase):
    def setUp(self):
        self.identificador = Identificador_de_rostros("../modelos/face-reidentification-retail-0095/face-reidentification-retail-0095.xml", "../modelos/face-reidentification-retail-0095/face-reidentification-retail-0095.bin", "CPU", 0.5)
        # Mocking super().procesar_frame()
        self.identificador.procesar_frame = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
        # Mocking self.choferes_dict
        self.identificador.choferes_dict = {
            "Chofer 1": [0.1, 0.2, 0.3, 0.4, 0.5],
            "Chofer 2": [0.2, 0.3, 0.4, 0.5, 0.6]
        }

    def test_procesar_frame(self):
        frame = MagicMock()
        vector = self.identificador.procesar_frame(frame)
        self.assertEqual(vector, [0.1, 0.2, 0.3, 0.4, 0.5])

    def test_obtener_nombre_conductor(self):
        frame = MagicMock()
        self.identificador.rostro = MagicMock()
        self.identificador.rostro.nombre = "DESCONOCIDO"
        self.identificador.confidence_threshold = 0.4
        # Mocking self.procesar_frame()
        self.identificador.procesar_frame = MagicMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5])
        self.identificador.obtener_nombre_conductor(frame)
        self.assertEqual(self.identificador.rostro.nombre, "Chofer 1")

class TestDetectorRasgosFaciales(unittest.TestCase):
    def setUp(self):
        self.detector = Detector_rasgos_faciales("../modelos/shape_predictor_68_face_landmarks/shape_predictor_68_face_landmarks.dat")

    def test_convert_to_dlib_rect(self):
        self.frame = MagicMock()
        self.frame.shape = [100,100,100]
        rect = self.detector.convert_to_dlib_rect(self.frame)
        self.assertEqual(rect.left(), 15)
        self.assertEqual(rect.top(), 20)
        self.assertEqual(rect.width(), self.frame.shape[1] * 0.71)  # scale factor 0.15
        self.assertEqual(rect.height(), self.frame.shape[0] * 0.81)  # scale factor 0.20

    def test_calcular_relacion_de_aspecto(self):
        obj = [[0, 0], [1, 0], [0, 1], [1, 1], [0, 0.5], [1, 0.5]]
        ear = self.detector.calcular_relacion_de_aspecto(obj)
        self.assertAlmostEqual(ear, 0.35, 2)

    def test_calcular_ear(self):
        rostro = MagicMock()
        rostro.ojo_izquierdo = [[0, 0], [1, 0], [0, 1], [1, 1], [0, 0.5], [1, 0.5]]
        rostro.ojo_derecho = rostro.ojo_izquierdo
        self.detector.calcular_ear(rostro)
        self.assertEqual(self.detector.v_pestaneo, 0)
        self.assertEqual(self.detector.contador_ojos_cerrados, 0)
        self.assertAlmostEqual(self.detector.tiempo_pestaneo, 0.0)

    def test_calcular_mar(self):
        rostro = MagicMock()
        rostro.boca = [[0, 0], [1, 0], [0, 1], [1, 1], [0, 0.5], [1, 0.5]]
        self.detector.calcular_mar(rostro)
        self.assertEqual(self.detector.contador_bostezos, 0)

    def test_calcular_nivel(self):
        nivel = self.detector.calcular_nivel(10)
        self.assertEqual(nivel, 0)

    def test_calcular_somnolencia(self):
        self.detector.v_pestaneo = 1
        self.detector.tiempo_pestaneo = 1000  # ms
        self.detector.somnolencia = 10
        nivel = self.detector.calcular_somnolencia()
        self.assertEqual(nivel, 2)


if __name__ == '__main__':
    unittest.main()