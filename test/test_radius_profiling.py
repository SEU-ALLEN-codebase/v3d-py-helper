import unittest
from v3dpy.loaders import PBD
from v3dpy.neuron_utilities import swc_handler
from v3dpy.neuron_utilities import radius


class MyTestCase(unittest.TestCase):
    def test_something(self):
        swc = r"D:\rectify\manual\15257_13263_25518_2294.swc"
        img = r"D:\rectify\my\15257_13263_25518_2294.v3dpbd"
        img = PBD().load(img)[0]
        swc = swc_handler.parse_swc(swc)
        res = radius.neuron_radius(swc, img, True, -1)
        swc_handler.write_swc(res, 'profiled.swc')


if __name__ == '__main__':
    unittest.main()
