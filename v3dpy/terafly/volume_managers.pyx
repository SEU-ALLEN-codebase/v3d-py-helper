cimport cython
from pathlib import Path
from .config import *
import struct
import numpy as np
cimport numpy as cnp

from libcpp.string cimport string
from libcpp.vector cimport vector
from .format_managers cimport Tiff3DFmtMngr, VirtualFmtMngr
from libc.stdint cimport uint32_t, int64_t, int32_t, uint16_t


cdef class VirtualVolume:
    cdef:
        public object root_dir
        public float VXL_V, VXL_H, VXL_D
        public float ORG_V, ORG_H, ORG_D
        public uint32_t DIM_V, DIM_H, DIM_D, DIM_C
        public uint32_t BYTESxCHAN
    def __cinit__(self):
        self.ORG_V = self.ORG_H = self.ORG_D = 0.0
        self.DIM_V = self.DIM_H = self.DIM_D = self.DIM_C = 0
        self.BYTESxCHAN = 0

    def __init__(self, root_dir: Path = None, vxl_1=0, vxl_2=0, vxl_3=0):
        self.root_dir = root_dir
        self.VXL_V = vxl_1
        self.VXL_H = vxl_2
        self.VXL_D = vxl_3


    cpdef cnp.ndarray load_sub_volume(self, int32_t v0=-1, int32_t v1=-1, int32_t h0=-1,
                                    int32_t h1=-1, int32_t d0=-1, int32_t d1=-1):
        raise NotImplementedError



cdef struct Rect:
    int32_t H0
    int32_t H1
    int32_t V0
    int32_t V1


cdef struct Segm:
    uint32_t D0
    uint32_t D1
    uint32_t ind0
    uint32_t ind1


cdef class Block(VirtualVolume):
    cdef:
        public str DIR_NAME
        public object CONTAINER
        public uint16_t ROW_INDEX, COL_INDEX
        public uint32_t HEIGHT, WIDTH, DEPTH, N_BLOCKS, N_CHANS, N_BYTESxCHAN
        public int32_t ABS_V, ABS_H
        public vector[string] FILENAMES
        public vector[uint32_t] BLOCK_SIZE
        public vector[int32_t] BLOCK_ABS_D

    def __cinit__(self):
        self.BLOCK_SIZE = vector[uint32_t]()
        self.BLOCK_ABS_D = vector[int32_t]()
        self.FILENAMES = vector[string]()
        self.HEIGHT = self.WIDTH = self.DEPTH = self.N_BLOCKS = self.N_CHANS \
            = self.ABS_V = self.ABS_H = 0
        self.DIR_NAME= ''
        self.N_BYTESxCHAN = 0

    def __init__(self, object container, uint16_t row_index, uint16_t col_index, object file):
        super(Block, self).__init__()
        self.CONTAINER = container
        self.ROW_INDEX = row_index
        self.COL_INDEX = col_index
        self.unbinarize_from(file)

    cdef void unbinarize_from(self, object file):
        cdef uint32_t i
        cdef uint16_t str_size
        self.HEIGHT, self.WIDTH, self.DEPTH, self.N_BLOCKS, self.N_CHANS, self.ABS_V, self.ABS_H, str_size = \
            struct.unpack('IIIIIiiH', file.read(30))
        self.DIR_NAME = file.read(str_size).decode('utf-8').rstrip("\x00")
        for i in range(self.N_BLOCKS):
            str_size = struct.unpack('H', file.read(2))[0]
            self.FILENAMES.push_back(file.read(str_size).rstrip(b"\x00"))
            self.BLOCK_SIZE.push_back(struct.unpack('I', file.read(4))[0])
            self.BLOCK_ABS_D.push_back(struct.unpack('i', file.read(4))[0])
        self.N_BYTESxCHAN = struct.unpack('I', file.read(4))[0]

    @cython.boundscheck(False)
    @cython.nonecheck(False)
    cpdef bytes get_fmt(self):
        cdef bytes temp = bytes(self.FILENAMES[0]).split(b'.')[-1]
        if temp == b'tif' or temp == b'tif':
            return b'Tiff3D'
        elif temp == b'v3draw':
            return b'Vaa3DRaw'
        else:
            raise IOError(f'Block: Unknown file format {temp}')


cdef bint intersects_segm(Block block, int32_t d0, int32_t d1, Segm& out):
    cdef int32_t i0 = 0, i1 = block.N_BLOCKS - 1

    if d0 >= <int32_t>(block.BLOCK_ABS_D[i1] + block.BLOCK_SIZE[i1]) or d1 <= 0:
        return False # no intersection

    while i0 < i1:
        if d0 < block.BLOCK_ABS_D[i0 + 1]:
            break
        i0 += 1

    while i1 > 0:
        if d1 > block.BLOCK_ABS_D[i1]:
            break
        i1 -= 1

    out.D0 = max(d0, 0)
    out.D1 = min(d1, <int32_t>block.DEPTH)
    out.ind0 = i0
    out.ind1 = i1

    return True

cdef bint intersects_rect(Block block, const Rect& area, Rect& out):
    cdef:
        int32_t t1 = block.ABS_H + block.WIDTH
        int32_t t2 = block.ABS_V + block.HEIGHT

    if area.H0 < t1 and area.H1 > block.ABS_H and area.V0 < t2 and area.V1 > block.ABS_V:

        out.H0 = max(block.ABS_H, area.H0)
        out.V0 = max(block.ABS_V, area.V0)
        out.H1 = min(t1, area.H1)
        out.V1 = min(t2, area.V1)

        return True
    else:
        return False


cdef class TiledVolume(VirtualVolume):
    cdef:
        uint16_t N_ROWS, N_COLS  # <-- Static type declarations
        list BLOCKS  # <-- Declare a 2D C-style array of Blocks
        int32_t reference_system_first, reference_system_second, reference_system_thrid
        float VXL_1, VXL_2, VXL_3

    def __cinit__(self):
        self.reference_system_first = self.reference_system_second = self.reference_system_thrid = \
            self.VXL_1 = self.VXL_2 = self.VXL_3 = self.N_ROWS = self.N_COLS = 0

    def __init__(self, root_dir: Path):
        super(TiledVolume, self).__init__(root_dir)
        self.BLOCKS = None
        mdata_filepath = root_dir / MDATA_BIN_FILE_NAME
        if mdata_filepath.is_file():  # We need to convert string back to Path object for is_file()
            self.load(mdata_filepath)
            self.init_channels()
        else:
            raise ValueError(f"TiledVolume: unable to find metadata file at {mdata_filepath}")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    cdef load(self, mdata_filepath: Path):
        cdef uint16_t i, j

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
    cpdef cnp.ndarray load_sub_volume(self, int32_t V0=-1, int32_t V1=-1, int32_t H0=-1,
                                      int32_t H1=-1, int32_t D0=-1, int32_t d1=-1):
        cdef:
            uint16_t row, col
            int32_t k, sV0, sV1, sH0, sH1, sD0, sD1, bV0, bH0, bD0
            Block block
            Rect intersect_area
            Segm intersect_segm
            bytes slice_fullpath
            bint first_time = True
            bytes ffmt

        V0, H0, D0 = max(0, V0), max(0, H0), max(0, D0)
        V1 = V1 if 0 <= V1 <= <int32_t>self.DIM_V else self.DIM_V
        H1 = H1 if 0 <= H1 <= <int32_t>self.DIM_H else self.DIM_H
        d1 = d1 if 0 <= d1 <= <int32_t>self.DIM_D else self.DIM_D
        assert V1 > V0 and H1 > H0 and d1 > D0, "TiledVolume: The start position should be lower than the end position."

        cdef:
            int64_t sbv_height = V1 - V0
            int64_t sbv_width = H1 - H0
            int64_t sbv_depth = d1 - D0
            int64_t sbv_channels
            int64_t sbv_bytes_chan
            cnp.ndarray npsubvol
            unsigned char* subvol
            VirtualFmtMngr fmt_mngr

        subvol_area = Rect()
        subvol_area.H0 = H0
        subvol_area.H1 = H1
        subvol_area.V0 = V0
        subvol_area.V1 = V1

        ffmt = self.BLOCKS[0][0].get_fmt()
        if ffmt == b'Tiff3D':
            fmt_mngr = Tiff3DFmtMngr()
        elif ffmt == b'Vaa3DRaw':
            raise NotImplementedError

        if intersects_segm(self.BLOCKS[0][0], D0, d1, intersect_segm):
            for row in range(self.N_ROWS):
                for col in range(self.N_COLS):
                    block = self.BLOCKS[row][col]
                    if intersects_rect(block, subvol_area, intersect_area):
                        for k in range(intersect_segm.ind0, intersect_segm.ind1 + 1):
                            if first_time:
                                first_time = False
                                sbv_channels = self.DIM_C
                                sbv_bytes_chan = self.BYTESxCHAN
                                assert sbv_channels == 1 , "TiledVolume: Multi channel not supported yet."
                                if sbv_bytes_chan == 1:
                                    dt = np.uint8
                                elif sbv_bytes_chan == 2:
                                    dt = np.uint16
                                elif sbv_bytes_chan == 4:
                                    dt = np.float32
                                else:
                                    raise ValueError(f"TiledVolume: Unsupported Datatype {self.BYTESxCHAN}")
                                npsubvol = np.zeros((sbv_channels, sbv_depth, sbv_height, sbv_width), dtype=dt)
                                subvol = <unsigned char *> npsubvol.data

                            slice_fullpath = str(self.root_dir / block.DIR_NAME /
                                                 bytes(block.FILENAMES[k]).decode('utf-8')).encode('utf-8')

                            # vertices of file block
                            sV0 = 0 if V0 < intersect_area.V0 else V0 - block.ABS_V
                            sV1 = block.HEIGHT if V1 > intersect_area.V1 else V1 - block.ABS_V
                            sH0 = 0 if H0 < intersect_area.H0 else H0 - block.ABS_H
                            sH1 = block.WIDTH if H1 > intersect_area.H1 else H1 - block.ABS_H
                            sD0 = 0 if D0 < block.BLOCK_ABS_D[k] else D0 - block.BLOCK_ABS_D[k]
                            sD1 = block.BLOCK_SIZE[k] if d1 > <int32_t>(block.BLOCK_ABS_D[k] + block.BLOCK_SIZE[k]) \
                                else d1 - block.BLOCK_ABS_D[k]

                            # vertices of buffer block
                            bV0 = 0 if V0 > intersect_area.V0 else intersect_area.V0 - V0
                            bH0 = 0 if H0 > intersect_area.H0 else intersect_area.H0 - H0
                            bD0 = 0 if D0 > block.BLOCK_ABS_D[k] else block.BLOCK_ABS_D[k] - D0

                            if b"NULL.tif" in slice_fullpath:
                                continue

                            fmt_mngr.copy_file_block2buffer(slice_fullpath,
                                                            sV0, sV1, sH0, sH1, sD0, sD1,
                                                            subvol, sbv_bytes_chan,
                                                            bH0 + bV0 * sbv_width + bD0 * sbv_width * sbv_height,
                                                            sbv_width,
                                                            sbv_width * sbv_height,
                                                            sbv_width * sbv_height * sbv_depth)
        else:
            raise IOError("TiledVolume: Depth interval out of range")

        return npsubvol
