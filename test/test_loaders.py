import unittest
from v3dpy.loaders.pbd import PBD
from v3dpy.loaders.raw import Raw
from pathlib import Path
import numpy as np

test_data_path = Path('d:/')


class LoaderTest(unittest.TestCase):

    def test_raw8(self):
        loader = Raw()
        in_img = loader.load(test_data_path / '8.v3draw')
        loader.save(test_data_path / '8_.v3draw', in_img)
        out_img = loader.load(test_data_path / '8_.v3draw')
        self.assertEqual((in_img == out_img).all(), True)  # add assertion here

    def test_raw16(self):
        loader = Raw()
        in_img = loader.load(test_data_path / '16.v3draw')
        loader.save(test_data_path / '16_.v3draw', in_img)
        out_img = loader.load(test_data_path / '16_.v3draw')
        self.assertEqual((in_img == out_img).all(), True)  # add assertion here

    def test_pbd8(self):
        loader = PBD()
        in_img = loader.load(test_data_path / '8.v3dpbd')
        loader.save(test_data_path/ '8_.v3dpbd', in_img)
        out_img = loader.load(test_data_path / '8_.v3dpbd')
        self.assertEqual((in_img == out_img).all(), True)  # add assertion here

    def test_pbd16_dbg(self):
        loader = PBD(pbd16_full_blood=False)
        in_img = loader.load(test_data_path / '16.v3dpbd')
        loader.save(test_data_path / '16_.v3dpbd', in_img)
        out_img = loader.load(test_data_path / '16_.v3dpbd')
        self.assertEqual((in_img == out_img).all(), True)  # add assertion here

    def test_pbd16_fb(self):
        loader = PBD(pbd16_full_blood=True)
        in_img = loader.load(test_data_path / '16.v3dpbd')
        loader.save(test_data_path / '16_.v3dpbd', in_img)
        out_img = loader.load(test_data_path / '16_.v3dpbd')
        self.assertEqual((in_img == out_img).all(), True)  # add assertion here

    def test_pbd16_new(self):
        loader = PBD()
        in_img = loader.load(test_data_path / '16.v3dpbd')
        loader.save16_halving(test_data_path / '16_.v3dpbd', in_img)
        # out_img = loader.load(test_data_path / 'pbd16.v3dpbd')
        # self.assertEqual((in_img == out_img).all(), True)  # add assertion here


if __name__ == '__main__':
    unittest.main()
