# Import relevant Cython modules
import numpy as np
cimport cython
from .tiff_manage cimport read_tiff_3d_file_to_buffer, load_tiff3d2metadata, close_tiff3d_file
from libc.stdint cimport uint8_t, int64_t, uint16_t, int32_t, uint32_t
from libc.string cimport memcpy
cimport numpy as cnp


cdef class VirtualFmtMngr:
    cdef void copy_file_block2buffer(self, const char* filename, int32_t sV0, int32_t sV1, int32_t sH0, int32_t sH1, int32_t sD0, int32_t sD1,
                                     uint8_t * buf, uint32_t pxl_size, int64_t offs, int64_t stridex,
                                     int64_t stridexy, int64_t stridexyz):
        raise NotImplementedError

    cdef void copy_block2sub_buf(self, uint8_t* src, uint8_t* dst,
                           int32_t dimi, int32_t dimj, int32_t dimk, int32_t typesize,
                           int64_t s_stridej, int64_t s_strideij,
                           int64_t d_stridej, int64_t d_strideij):

        cdef int32_t i, k, j
        cdef int64_t s_index, d_index
        cdef uint8_t* s_slice = src
        cdef uint8_t* d_slice = dst
        cdef uint8_t * s_stripe
        cdef uint8_t * d_stripe

        s_stridej *= typesize
        s_strideij *= typesize
        d_stridej *= typesize
        d_strideij *= typesize

        for k in range(dimk):
            s_stripe = s_slice
            d_stripe  = d_slice
            for i in range(dimi):
                memcpy(d_stripe, s_stripe, dimj * typesize)
                s_stripe += s_stridej
                d_stripe += d_stridej
            s_slice += s_strideij
            d_slice += d_strideij

    cdef void copy_rgb_block2vaa3d_raw_sub_buf(self, uint8_t* src, uint8_t* dst,
                                                 int32_t dimi, int32_t dimj, int32_t dimk, int32_t typesize,
                                                 int64_t s_stridej, int64_t s_strideij,
                                                 int64_t d_stridej, int64_t d_strideij, int64_t d_strideijk):

        cdef:
            int32_t c, i, j, k
            int64_t s_index, d_index
            uint8_t * s_slice
            uint8_t * d_slice
            uint8_t * s_stripe
            uint8_t * d_stripe
            uint16_t * src16
            uint16_t * dst16
            uint16_t * s_slice16
            uint16_t * d_slice16
            uint16_t * s_stripe16
            uint16_t * d_stripe16

        if typesize == 1:
            for c in range(3):
                s_slice = src
                d_slice = dst
                for k in range(dimk):
                    s_stripe = s_slice
                    d_stripe = d_slice
                    for i in range(dimi):
                        for j in range(dimj):
                            d_stripe[j] = s_stripe[j * 3]
                        s_stripe += s_stridej * 3
                        d_stripe += d_stridej
                    s_slice += s_strideij * 3
                    d_slice += d_strideij
                src += 1
                dst += d_strideijk
        else:  # must be guaranteed that typesize == 2
            for c in range(3):
                s_slice16 = <uint16_t*>src
                d_slice16 = <uint16_t*>dst
                for k in range(dimk):
                    s_stripe16 = s_slice16
                    d_stripe16 = d_slice16
                    for i in range(dimi):
                        for j in range(dimj):
                            d_stripe16[j] = s_stripe16[j * 3]
                        s_stripe16 += s_stridej * 3
                        d_stripe16 += d_stridej
                    s_slice16 += s_strideij * 3
                    d_slice16 += d_strideij
                src16 += 1
                dst16 += d_strideijk



cdef class Tiff3DFmtMngr(VirtualFmtMngr):

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void copy_file_block2buffer(self, const char* filename,
                                     int32_t sV0, int32_t sV1, int32_t sH0, int32_t sH1, int32_t sD0, int32_t sD1,
                                     uint8_t* buf, uint32_t pxl_size,
                                     int64_t offs,
                                     int64_t stridex,
                                     int64_t stridexy,
                                     int64_t stridexyz):
        cdef:
            unsigned int sz[4]
            int datatype = 0
            bint b_swap = 0
            void* fhandle = load_tiff3d2metadata(filename, sz[0], sz[1], sz[2], sz[3], datatype, b_swap)

        if datatype != pxl_size:
            raise IOError("Tiff3DFmtMngr.copy_file_block2buffer: source data type differs from destination pixel size.")

        cdef cnp.ndarray[cnp.uint8_t, ndim=1] npbuf_t = np.zeros(sz[0] * sz[1] * (sD1 - sD0) * sz[3] * datatype, dtype=np.uint8)
        cdef unsigned char * buf_t = <uint8_t *> npbuf_t.data

        read_tiff_3d_file_to_buffer(fhandle, buf_t, sz[0], sz[1], sD0, sD1 - 1, b_swap,1, -1, -1, -1, -1)
        close_tiff3d_file(fhandle)

        cdef:
            int64_t s_stridej = sz[0]
            int64_t s_strideij = sz[0] * sz[1]
            int64_t d_stridej = stridex
            int64_t d_strideij = stridexy
            int64_t d_strideijk = stridexyz
            int32_t dimi = sV1 - sV0
            int32_t dimj = sH1 - sH0
            int32_t dimk = sD1 - sD0

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