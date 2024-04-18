import unittest
from unittest.mock import MagicMock
from objetos import Frame, Rostro
from motores_de_inferencia import Detector_de_rostros, Motor_de_inferencia, Identificador_de_rostros

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

if __name__ == '__main__':
    unittest.main()