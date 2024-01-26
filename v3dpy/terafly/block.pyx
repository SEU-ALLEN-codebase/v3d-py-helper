from .virtual_volume import VirtualVolume
import cython
import struct
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef class Rect:
    cdef long long H0, H1, V0, V1  # Use 'public' to make attributes accessible in Python-space

    def __init__(self, int h0, int v0, int h1, int v1):
        self.H0 = h0
        self.H1 = h1
        self.V0 = v0
        self.V1 = v1


cdef class Segm:
    cdef long long D0, D1, ind0, ind1  # Use 'public' to make attributes accessible in Python-space

    def __init__(self, int d0, int d1, int ind0, int ind1):
        self.D0 = d0
        self.D1 = d1
        self.ind0 = ind0
        self.ind1 = ind1


cdef class Block(VirtualVolume):
    cdef object CONTAINER
    cdef long long ROW_INDEX, COL_INDEX
    cdef str DIR_NAME
    cdef long long HEIGHT, WIDTH, DEPTH, N_BLOCKS, N_CHANS, N_BYTESxCHAN, ABS_V, ABS_H
    cdef vector[string] FILENAMES
    cdef vector[long long] BLOCK_SIZE
    cdef vector[long long] BLOCK_ABS_D

    def __init__(self, container, long long row_index, long long col_index, file):
        super(Block, self).__init__()
        self.BLOCK_SIZE = NULL
        self.BLOCK_ABS_D = NULL
        self.FILENAMES = NULL
        self.CONTAINER = container
        self.ROW_INDEX = row_index
        self.COL_INDEX = col_index
        self.unbinarize_from(file)

    cdef void unbinarize_from(self, object file):
        cdef long long i, str_size
        self.HEIGHT, self.WIDTH, self.DEPTH, self.N_BLOCKS, self.N_CHANS, self.ABS_V, self.ABS_H, str_size = \
            struct.unpack('IIIIIiiH', file.read(30))
        self.DIR_NAME = file.read(str_size).decode('utf-8').rstrip("\x00")
        for i in range(self.N_BLOCKS):
            str_size = struct.unpack('H', file.read(2))[0]
            self.FILENAMES.push_back(file.read(str_size).rstrip(b"\x00"))
            self.BLOCK_SIZE.push_back(struct.unpack('I', file.read(4))[0])
            self.BLOCK_ABS_D.push_back(struct.unpack('i', file.read(4))[0])
        self.N_BYTESxCHAN = struct.unpack('I', file.read(4))[0]

    cdef Segm intersects_segm(self, int d0, int d1):
        cdef int i0 = 0
        cdef int i1 = self.N_BLOCKS - 1

        if d0 >= self.BLOCK_ABS_D[i1] + self.BLOCK_SIZE[i1] or d1 <= 0:
            return None # no intersection

        while i0 < i1:
            if d0 < self.BLOCK_ABS_D[i0 + 1]:
                break
            i0 += 1

        while i1 > 0:
            if d1 > self.BLOCK_ABS_D[i1]:
                break
            i1 -= 1

        return Segm(max(d0, 0), min(d1, self.DEPTH), i0, i1)

    cdef Rect intersects_rect(self, Rect area):
        cdef int abs_h_plus_width = self.ABS_H + self.WIDTH
        cdef int abs_v_plus_height = self.ABS_V + self.HEIGHT

        if (area.H0 < abs_h_plus_width and area.H1 > self.ABS_H and
            area.V0 < abs_v_plus_height and area.V1 > self.ABS_V):

            return Rect(
                max(self.ABS_H, area.H0),
                max(self.ABS_V, area.V0),
                min(abs_h_plus_width, area.H1),
                min(abs_v_plus_height, area.V1)
            )
        else:
            return None

    cdef str get_fmt(self):
        temp = bytes(self.FILENAMES[0]).decode('utf-8').split('.')
        if temp == 'tif' or temp == 'tif':
            return 'Tiff3D'
        elif temp == 'v3draw':
            return 'Vaa3DRaw'
        else:
            raise IOError(f'Block: Unknown file format {temp}')