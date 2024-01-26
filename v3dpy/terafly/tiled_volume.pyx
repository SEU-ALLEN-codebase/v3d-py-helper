cimport cython
from .virtual_volume import VirtualVolume
from pathlib import Path
from .config import *
import struct
from .block import Block, Rect
import numpy as np
from libc.stdlib cimport malloc, free

ctypedef np.uint8_t uint8
ctypedef np.uint16_t uint16
ctypedef np.float32_t float32


cdef class TiledVolume(VirtualVolume):
    cdef long long N_ROWS, N_COLS  # <-- Static type declarations
    cdef Block** BLOCKS  # <-- Declare a 2D C-style array of Blocks

    def __init__(self, root_dir: Path):
        super().__init__(root_dir)
        self.BLOCKS = NULL
        mdata_filepath = root_dir / MDATA_BIN_FILE_NAME
        if mdata_filepath.is_file():  # We need to convert string back to Path object for is_file()
            self.load(mdata_filepath)
            self.init_channels()
        else:
            raise ValueError(f"in TiledVolume: unable to find metadata file at {mdata_filepath}")

    def __dealloc__(self):  # Don't forget to free your memory!
        if self.BLOCKS is not NULL:
            for i in range(self.N_ROWS):
                if self.BLOCKS[i] is not NULL:
                    free(self.BLOCKS[i])
            free(self.BLOCKS)

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

            self.BLOCKS = <Block **> malloc(self.N_ROWS * sizeof(Block *))  # Allocate space for row pointers
            if self.BLOCKS is NULL:
                raise MemoryError()
            for i in range(self.N_ROWS):
                self.BLOCKS[i] = <Block *> malloc(self.N_COLS * sizeof(Block))  # Allocate space for each row
                if self.BLOCKS[i] is NULL:
                    raise MemoryError()

            # Initialize blocks
            for i in range(self.N_ROWS):
                for j in range(self.N_COLS):
                    self.BLOCKS[i][j] = Block(self, i, j, f)

    cdef init_channels(self):
        self.DIM_C = self.BLOCKS[0][0].N_CHANS
        self.BYTESxCHAN = self.BLOCKS[0][0].N_BYTESxCHAN

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray load_subvolume(self, long v0=-1, long v1=-1, long h0=-1, long h1=-1, long d0=-1, long d1=-1):
        cdef long long row, col, k, sv0, sv1, sh0, sh1, sd0, sd1, bv0, bh0, bd0
        cdef Block block
        cdef Rect intersect_area
        cdef str dt
        cdef np.ndarray buffer
        cdef Path slice_fullpath
        cdef np.ndarray img
        cdef bint first_time = True

        v0, h0, d0 = max(0, v0), max(0, h0), max(0, d0)
        v1 = v1 if 0 <= v1 <= self.DIM_V else self.DIM_V
        h1 = h1 if 0 <= h1 <= self.DIM_H else self.DIM_H
        d1 = d1 if 0 <= d1 <= self.DIM_D else self.DIM_D
        assert v1 > v0 and h1 > h0 and d1 > d0
        subvol_area = Rect(h0, v0, h1, v1)
        intersect_segm = self.BLOCKS[0][0].intersects_segm(d0, d1)
        ffmt = self.BLOCKS[0][0].get_fmt()

        if intersect_segm is not None:
            for row in range(self.N_ROWS):
                for col in range(self.N_COLS):
                    block = self.BLOCKS[row][col]
                    intersect_area = block.intersects_rect(subvol_area)

                    if intersect_area is not None:
                        for k in range(intersect_segm.ind0, intersect_segm.ind1 + 1):
                            if first_time:
                                first_time = False
                                assert self.DIM_C == 1  # multi channel not supported here
                                if self.BYTESxCHAN == 1:
                                    dt = np.uint8
                                elif self.BYTESxCHAN == 2:
                                    dt = np.uint16
                                elif self.BYTESxCHAN == 4:
                                    dt = np.float32
                                else:
                                    raise ValueError(f"Unsupported Datatype {self.BYTESxCHAN}")
                                buffer = np.zeros((d1 - d0, v1 - v0, h1 - h0), dtype=dt)

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


                            buffer[bd0:bd0 + sd1 - sd0, bv0:bv0 + sv1 - sv0, bh0:bh0 + sh1 - sh0] = img[sd0:sd1,
                                                                                                    sv0:sv1, sh0:sh1]
        else:
            raise RuntimeError("TiledVolume: Depth interval out of range")

        return buffer
