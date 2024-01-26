# Import relevant Cython modules
import cython

cdef class VirtualVolume:
    cdef long long VXL_V, VXL_H, VXL_D
    cdef str root_dir
    cdef float ORG_V, ORG_H, ORG_D
    cdef long long DIM_V, DIM_H, DIM_D, DIM_C
    cdef long long BYTESxCHAN
    cdef long long t0, t1
    cdef long long DIM_T

    def __init__(self, root_dir='', vxl_1=0, vxl_2=0, vxl_3=0):
        self.root_dir = root_dir
        self.VXL_V = vxl_1
        self.VXL_H = vxl_2
        self.VXL_D = vxl_3
        self.ORG_V = self.ORG_H = self.ORG_D = 0.0
        self.DIM_V = self.DIM_H = self.DIM_D = self.DIM_C = 0
        self.BYTESxCHAN = 0
        self.t0 = self.t1 = 0
        self.DIM_T = 1