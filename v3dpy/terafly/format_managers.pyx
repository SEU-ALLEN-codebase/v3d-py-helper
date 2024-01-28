# Import relevant Cython modules
import numpy as np
cimport numpy as np
cimport cython
from libcpp.string cimport string
from tiff_manage import read_tiff_3d_file_to_buffer, load_tiff3d2metadata, close_tiff3d_file
from libc.stdint cimport uint8_t, int64_t
cimport numpy as cnp


cdef class VirtualFmtMngr:
    cdef void copy_file_block2buffer(self, string filename, int sV0, int sV1, int sH0, int sH1, int sD0, int sD1,
                                     np.uint8_t * buf, int pxl_size, np.int64_t offs, np.int64_t stridex,
                                     np.int64_t stridexy, np.int64_t stridexyz):
        raise NotImplementedError

    def copy_block2sub_buf(self, np.uint8_t* src, np.uint8_t* dst,
                           int dimi, int dimj, int dimk, int typesize,
                           np.int64_t s_stridej, np.int64_t s_strideij,
                           np.int64_t d_stridej, np.int64_t d_strideij):

        cdef int i, k, j
        cdef np.int64_t s_index, d_index

        s_stridej *= typesize
        s_strideij *= typesize
        d_stridej *= typesize
        d_strideij *= typesize

        for k in range(dimk):
            for i in range(dimi):
                s_index = k * s_strideij + i * s_stridej
                d_index = k * d_strideij + i * d_stridej
                for j in range(dimj * typesize):
                    dst[d_index + j] = src[s_index + j]

    def copy_rgb_block2vaa3d_raw_sub_buf(self, np.uint8_t* src, np.uint8_t* dst,
                                         int dimi, int dimj, int dimk, int typesize,
                                         np.int64_t s_stridej, np.int64_t s_strideij,
                                         np.int64_t d_stridej, np.int64_t d_strideij, np.int64_t d_strideijk):

        cdef int c, i, j, k
        cdef np.int64_t s_index, d_index
        cdef np.uint16_t * src16 = <np.uint16_t *> src
        cdef np.uint16_t * dst16 = <np.uint16_t *> dst

        if typesize == 1:
            for c in range(3):
                for k in range(dimk):
                    for i in range(dimi):
                        s_index = c + k * 3 * s_strideij + i * 3 * s_stridej
                        d_index = c * d_strideijk + k * d_strideij + i * d_stridej
                        for j in range(dimj):
                            dst[d_index + j] = src[s_index + 3 * j]
                src = src + 1
                dst = dst + d_strideijk
        else:  # must be guaranteed that typesize == 2
            for c in range(3):
                for k in range(dimk):
                    for i in range(dimi):
                        s_index = c + k * 3 * s_strideij + i * 3 * s_stridej
                        d_index = c * d_strideijk + k * d_strideij + i * d_stridej
                        for j in range(dimj):
                            dst16[d_index + j] = src16[s_index + 3 * j]
                src16 = src16 + 1
                dst16 = dst16 + d_strideijk


@cython.boundscheck(False)
@cython.wraparound(False)
cdef class Tiff3DFmtMngr(VirtualFmtMngr):

    cdef void copy_file_block2buffer(self, string filename, int sV0, int sV1, int sH0, int sH1, int sD0, int sD1,
                                    np.uint8_t* buf, int pxl_size, np.int64_t offs, np.int64_t stridex,
                                    np.int64_t stridexy, np.int64_t stridexyz):
        cdef:
            unsigned int sz[4]
            int datatype = 0
            bint b_swap = 0
            unsigned long long fhandle = load_tiff3d2metadata(filename, sz[0], sz[1], sz[2], sz[3], datatype, b_swap)

        if datatype != pxl_size:
            raise IOError("Tiff3DFmtMngr.copy_file_block2buffer: source data type differs from destination pixel size.")

        cdef cnp.ndarray[cnp.uint8_t, ndim=1] npbuf_t = np.zeros(sz[0] * sz[1] * (sD1 - sD0) * sz[3] * datatype, dtype=np.uint8)
        cdef unsigned char * buf_t = <uint8_t *> npbuf_t.data

        read_tiff_3d_file_to_buffer(fhandle, buf_t, sz[0], sz[1], sD0, sD1 - 1, b_swap)
        close_tiff3d_file(fhandle)

        cdef:
            int64_t s_stridej = sz[0]
            int64_t s_strideij = sz[0] * sz[1]
            int64_t d_stridej = stridex
            int64_t d_strideij = stridexy
            int64_t d_strideijk = stridexyz


            int dimi = sV1 - sV0
            int dimj = sH1 - sH0
            int dimk = sD1 - sD0

        if sz[3] == 1:  # single channel Tiff
            self.copy_block2sub_buf(
                buf_t + pxl_size * (s_stridej * sV0 + sH0),
                buf + pxl_size * offs,
                dimi, dimj, dimk, pxl_size,
                s_stridej, s_strideij, d_stridej, d_strideij
            )
        elif sz[3] == 3:  # RGB Tiff
            self.copy_rgb_block2vaa3d_raw_sub_buf(
                buf_t + 3 * pxl_size * (s_stridej * sV0 + sH0),
                buf + pxl_size * offs,
                dimi, dimj, dimk, pxl_size,
                s_stridej, s_strideij,
                d_stridej, d_strideij, d_strideijk)
        else:
            raise IOError("Tiff3DFmtMngr.copy_file_block2buffer: unsupported number of channels.")