from libc.stdint cimport uint8_t, int64_t, int32_t, uint32_t


cdef class VirtualFmtMngr:
    cdef public void copy_file_block2buffer(self, const char* filename, int32_t sV0, int32_t sV1, int32_t sH0, int32_t sH1, int32_t sD0, int32_t sD1,
                                             uint8_t * buf, uint32_t pxl_size, int64_t offs, int64_t stridex,
                                             int64_t stridexy, int64_t stridexyz)

    cdef void copy_block2sub_buf(self, uint8_t * src, uint8_t * dst,
                                int32_t dimi, int32_t dimj, int32_t dimk, int32_t typesize,
                                int64_t s_stridej, int64_t s_strideij,
                                int64_t d_stridej, int64_t d_strideij)

    cdef void copy_rgb_block2vaa3d_raw_sub_buf(self, uint8_t * src, uint8_t * dst,
                                              int32_t dimi, int32_t dimj, int32_t dimk, int32_t typesize,
                                              int64_t s_stridej, int64_t s_strideij,
                                              int64_t d_stridej, int64_t d_strideij, int64_t d_strideijk)


cdef class Tiff3DFmtMngr(VirtualFmtMngr):
    cdef public void copy_file_block2buffer(self, const char* filename, int32_t sV0, int32_t sV1, int32_t sH0, int32_t sH1, int32_t sD0, int32_t sD1,
                                            uint8_t* buf, uint32_t pxl_size, int64_t offs, int64_t stridex,
                                            int64_t stridexy, int64_t stridexyz)