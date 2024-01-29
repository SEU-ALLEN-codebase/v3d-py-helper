import unittest
from v3dpy.loaders import PBD
from v3dpy.terafly import TeraflyInterface
from pathlib import Path
import numpy as np


test_data_path = Path(r'D:\GitHub\TeraQC\data')
outdir = Path('testoutput')


class TeraflyTest(unittest.TestCase):

    def test_image_dims(self):
        for tfpath in test_data_path.rglob('RES*'):
            y, x, z = [int(i) for i in tfpath.name[4:-1].split('x')]
            t = TeraflyInterface(tfpath)
            self.assertTupleEqual(t.get_dim(), (x, y, z, 1), "Dims are not equal")

    def test_image_crop(self):
        for tfpath in test_data_path.rglob('RES*'):
            t = TeraflyInterface(tfpath)

            # center block
            size = np.array(t.get_dim()[:3])
            half_block_size = np.array([128, 128, 64]) // 2
            start = size // 2 - half_block_size
            end = size // 2 + half_block_size - 1

            img = t.get_sub_volume(start[0], end[0], start[1], end[1], start[2], end[2])
            # PBD(pbd16_full_blood=False).save(outdir / f'{tfpath.parent.name}.v3dpbd', img)


if __name__ == '__main__':
    unittest.main()
