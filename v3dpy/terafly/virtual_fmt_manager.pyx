# Import relevant Cython modules
import cython

cdef class VirtualFmtManager:
    cdef load_metadata(str fname):
        pass

    cdef close_file(fhandle):
        pass

    cdef load2sub_stack()