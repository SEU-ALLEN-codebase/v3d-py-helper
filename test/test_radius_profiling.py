import unittest
from v3dpy.loaders import PBD
from v3dpy.neuron_utilities import swc_handler
from v3dpy.neuron_utilities import radius
# import numpy as np

#
# def neuron_radius(tree, img, is2d, bkg_thr):
#     assert img.ndim == 3
#     tree = [list(t) for t in tree]
#     img_ = img.astype(np.float32)
#     for t in tree:
#         x, y, z = t[2:5]
#         t[5] = marker_radius_hanchuan_xy(x, y, z, img_, bkg_thr)
#     return [tuple(t) for t in tree]
#
#
# def marker_radius_hanchuan_xy(x, y, z, img, thr):
#     sz0 = img.shape[2]
#     sz1 = img.shape[1]
#     k = int(z)
#     max_r = sz0 / 2
#     max_r = min(max_r, sz1 / 2)
#     for ir in range(1, int(max_r + 1)):
#         total_num = background_num = 0
#
#         for dy in range(-ir, (ir + 1)):
#             for dx in range(-ir, (ir + 1)):
#                 total_num += 1
#                 r = (dx*dx + dy*dy) **.5
#                 if ir - 1 < r <= ir:
#                     i = int(x + dx)
#                     if i < 0 or i >= sz0:
#                         return ir
#                     j = int(y + dy)
#                     if j < 0 or j >= sz1:
#                         return ir
#                     if img[k, j, i] <= thr:
#                         background_num += 1
#                         if background_num / total_num > 0.001:
#                             return ir
#     return 1


class MyTestCase(unittest.TestCase):
    def test_something(self):
        swc = r"D:\rectify\manual\15257_13263_25518_2294.swc"
        img = r"D:\rectify\my\15257_13263_25518_2294.v3dpbd"
        img = PBD().load(img)[0]
        swc = swc_handler.parse_swc(swc)
        res = radius.neuron_radius(swc, img, True, 3)
        swc_handler.write_swc(res, 'profiled.swc')


if __name__ == '__main__':
    unittest.main()
