cimport cython
from pathlib import Path
from .config import *
import struct
import numpy as np
cimport numpy as cnp

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from libcpp.string cimport string
from libcpp.vector cimport vector
from format_managers import VirtualFmtMngr, Tiff3DFmtMngr


cdef class VirtualVolume:
    cdef long long VXL_V, VXL_H, VXL_D
    cdef str root_dir
    cdef float ORG_V, ORG_H, ORG_D
    cdef long long DIM_V, DIM_H, DIM_D, DIM_C
    cdef long long BYTESxCHAN
    cdef long long t0, t1
    cdef long long DIM_T

    def __init__(self, root_dir: Path, vxl_1=0, vxl_2=0, vxl_3=0):
        self.root_dir = root_dir
        self.VXL_V = vxl_1
        self.VXL_H = vxl_2
        self.VXL_D = vxl_3
        self.ORG_V = self.ORG_H = self.ORG_D = 0.0
        self.DIM_V = self.DIM_H = self.DIM_D = self.DIM_C = 0
        self.BYTESxCHAN = 0
        self.t0 = self.t1 = 0
        self.DIM_T = 1

    cpdef cnp.ndarray load_sub_volume(self, long long v0=-1, long long v1=-1, long long h0=-1,
                                    long long h1=-1, long long d0=-1, long long d1=-1):
        raise NotImplementedError



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
    DIR_NAME: str
    cdef object CONTAINER
    cdef long long ROW_INDEX, COL_INDEX
    cdef long long HEIGHT, WIDTH, DEPTH, N_BLOCKS, N_CHANS, N_BYTESxCHAN, ABS_V, ABS_H
    cdef vector[string] FILENAMES
    cdef vector[long long] BLOCK_SIZE
    cdef vector[long long] BLOCK_ABS_D

    def __init__(self, container, long long row_index, long long col_index, file):
        super(Block, self).__init__()
        self.BLOCK_SIZE = []
        self.BLOCK_ABS_D = []
        self.FILENAMES = []
        self.CONTAINER = container
        self.ROW_INDEX = row_index
        self.COL_INDEX = col_index
        self.HEIGHT = self.WIDTH = self.DEPTH = self.N_BLOCKS = self.N_CHANS = self.ABS_V = self.ABS_H = 0
        self.DIR_NAME= ''
        self.N_BYTESxCHAN = 0
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


cdef class TiledVolume(VirtualVolume):
    cdef long long N_ROWS, N_COLS  # <-- Static type declarations
    cdef list BLOCKS  # <-- Declare a 2D C-style array of Blocks

    def __init__(self, root_dir: Path):
        super(TiledVolume, self).__init__(root_dir)
        self.BLOCKS = None
        self.reference_system_first = self.reference_system_second = self.reference_system_thrid = \
            self.VXL_1 = self.VXL_2 = self.VXL_3 = self.N_ROWS = self.N_COLS = 0
        mdata_filepath = root_dir / MDATA_BIN_FILE_NAME
        if mdata_filepath.is_file():  # We need to convert string back to Path object for is_file()
            self.load(mdata_filepath)
            self.init_channels()
        else:
            raise ValueError(f"TiledVolume: unable to find metadata file at {mdata_filepath}")

    cdef load(self, mdata_filepath: Path):
        cdef long long i, j

        with open(mdata_filepath, 'rb') as f:
            mdata_version_read = struct.unpack('f', f.read(4))[0]
            if mdata_version_read != MDATA_BIN_FILE_VERSION:
                f.seek(0)
                str_size = struct.unpack('H', f.read(2))[0]
                f.read(str_size)
            self.reference_system_first, self.reference_system_second, self.reference_system_thrid, \
                self.VXL_1, self.VXL_2, self.VXL_3, self.VXL_V, self.VXL_H, self.VXL_D, \
                self.ORG_V, self.ORG_H, self.ORG_D, self.DIM_V, self.DIM_H, self.DIM_D, \
                self.N_ROWS, self.N_COLS = struct.unpack('iiifffffffffIIIHH', f.read(64))

            self.BLOCKS = [[Block(self, i, j, f) for j in range(self.N_COLS)] for i in range(self.N_ROWS)]

    cdef init_channels(self):
        self.DIM_C = self.BLOCKS[0][0].N_CHANS
        self.BYTESxCHAN = self.BLOCKS[0][0].N_BYTESxCHAN

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cpdef cnp.ndarray load_sub_volume(self, long long v0=-1, long long v1=-1, long long h0=-1,
                                    long long h1=-1, long long d0=-1, long long d1=-1):
        cdef long long row, col, k, sv0, sv1, sh0, sh1, sd0, sd1, bv0, bh0, bd0
        cdef Block block
        cdef Rect intersect_area
        cdef str dt
        cdef bint first_time = True

        v0, h0, d0 = max(0, v0), max(0, h0), max(0, d0)
        v1 = v1 if 0 <= v1 <= self.DIM_V else self.DIM_V
        h1 = h1 if 0 <= h1 <= self.DIM_H else self.DIM_H
        d1 = d1 if 0 <= d1 <= self.DIM_D else self.DIM_D
        assert v1 > v0 and h1 > h0 and d1 > d0

        cdef cnp.int64_t sbv_height = v1 - v0
        cdef cnp.int64_t sbv_width = h1 - h0
        cdef cnp.int64_t sbv_depth = d1 - d0
        cdef cnp.int64_t sbv_channels
        cdef cnp.int64_t sbv_bytes_chan
        cdef cnp.ndarray npsubvol
        cdef unsigned char * subvol

        subvol_area = Rect(h0, v0, h1, v1)
        intersect_segm = self.BLOCKS[0][0].intersects_segm(d0, d1)
        ffmt = self.BLOCKS[0][0].get_fmt()
        if ffmt == 'Tiff3D':
            fmt_mngr = Tiff3DFmtMngr()
        elif ffmt == 'Vaa3DRaw':
            raise NotImplementedError

        if intersect_segm is not None:
            for row in range(self.N_ROWS):
                for col in range(self.N_COLS):
                    block = self.BLOCKS[row][col]
                    intersect_area = block.intersects_rect(subvol_area)

                    if intersect_area is not None:
                        for k in range(intersect_segm.ind0, intersect_segm.ind1 + 1):
                            if first_time:
                                first_time = False
                                sbv_channels = self.DIM_C
                                sbv_bytes_chan = self.BYTESxCHAN
                                assert sbv_channels == 1  # multi channel not supported here
                                if sbv_bytes_chan == 1:
                                    dt = np.uint8
                                elif sbv_bytes_chan == 2:
                                    dt = np.uint16
                                elif sbv_bytes_chan == 4:
                                    dt = np.float32
                                else:
                                    raise ValueError(f"Unsupported Datatype {self.BYTESxCHAN}")
                                npsubvol = np.zeros((sbv_depth, sbv_height, sbv_width, sbv_channels), dtype=dt)
                                subvol = <unsigned char *> npsubvol.data

                            slice_fullpath = self.root_dir / block.DIR_NAME / bytes(block.FILENAMES[k]).decode('utf-8')

                            # vertices of file block
                            sv0 = 0 if v0 < intersect_area.V0 else v0 - block.ABS_V
                            sv1 = block.HEIGHT if v1 > intersect_area.V1 else v1 - block.ABS_V
                            sh0 = 0 if h0 < intersect_area.H0 else h0 - block.ABS_H
                            sh1 = block.WIDTH if h1 > intersect_area.H1 else h1 - block.ABS_H
                            sd0 = 0 if d0 < block.BLOCK_ABS_D[k] else d0 - block.BLOCK_ABS_D[k]
                            sd1 = block.BLOCK_SIZE[k] if d1 > block.BLOCK_ABS_D[k] + block.BLOCK_SIZE[k] \
                                else d1 - block.BLOCK_ABS_D[k]

                            # vertices of buffer block
                            bv0 = 0 if v0 > intersect_area.V0 else intersect_area.V0 - v0
                            bh0 = 0 if h0 > intersect_area.H0 else intersect_area.H0 - h0
                            bd0 = 0 if d0 > block.BLOCK_ABS_D[k] else block.BLOCK_ABS_D[k] - d0

                            if "NULL.tif" in str(slice_fullpath):
                                continue

                            fmt_mngr.copy_file_block2buffer(slice_fullpath, sv0, sv1, sh0, sh1, sd0, sd1,
                                                            subvol, <int>sbv_bytes_chan,
                                                            bh0 + bv0 * sbv_width + bd0 * sbv_width * sbv_height,
                                                            sbv_width,
                                                            sbv_width * sbv_height,
                                                            sbv_width * sbv_height * sbv_depth)
        else:
            raise IOError("TiledVolume: Depth interval out of range")

        return npsubvol
