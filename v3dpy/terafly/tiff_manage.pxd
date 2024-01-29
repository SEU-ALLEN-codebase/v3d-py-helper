cdef public void read_tiff_3d_file_to_buffer(void* fhandler,
                                             unsigned char * img,
                                             unsigned int img_width,
                                             unsigned int img_height,
                                             unsigned int first,
                                             unsigned int last,
                                             bint b_swap,
                                             int downsampling_factor,
                                             int start_i,
                                             int end_i,
                                             int start_j,
                                             int end_j)

cdef public void close_tiff3d_file(void* fhandle)

cdef public void* load_tiff3d2metadata(const char* filename, unsigned int& sz0, unsigned int& sz1, unsigned int& sz2,
                                       unsigned int& sz3, int& datatype, bint& b_swap)