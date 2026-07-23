import math
import unittest

from avaliar_deteccoes import casar, iou_3d


def caixa(classe="Box", centro=(0.0, 0.0, 1.0), dimensoes=(2.0, 1.0, 2.0), yaw=0.0):
    return {
        "classe": classe,
        "centro": list(centro),
        "dimensoes": list(dimensoes),
        "yaw_rad": yaw,
    }


class Avaliacao3DTest(unittest.TestCase):
    def test_iou_identica(self):
        self.assertAlmostEqual(iou_3d(caixa(), caixa()), 1.0)

    def test_iou_sem_intersecao(self):
        self.assertEqual(iou_3d(caixa(), caixa(centro=(3.0, 0.0, 1.0))), 0.0)

    def test_iou_rotacionada(self):
        valor = iou_3d(caixa(), caixa(yaw=math.pi / 4))
        self.assertGreater(valor, 0.5)
        self.assertLess(valor, 1.0)

    def test_casamento_um_para_um(self):
        pares, falsos_positivos, falsos_negativos = casar(
            [caixa()], [caixa()], threshold=0.5, mapa_classes=None
        )
        self.assertEqual(len(pares), 1)
        self.assertFalse(falsos_positivos)
        self.assertFalse(falsos_negativos)


if __name__ == "__main__":
    unittest.main()
