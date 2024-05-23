cimport numpy as cnp
import numpy as np
from libc.math cimport sqrt
import cython


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef object neuron_radius(object tree,  img, bint is2d, double bkg_thr):
    assert img.ndim == 3
    tree = [list(t) for t in tree]
    cdef float x, y, z, r
    cdef cnp.ndarray[cnp.float32_t, ndim=3] img_ = img.astype(np.float32)
    for t in tree:
        x, y, z = t[2:5]
        if is2d:
            t[5] = marker_radius_hanchuan_xy(x, y, z, img, bkg_thr)
        else:
            t[5] = marker_radius_hanchuan(x, y, z, img, bkg_thr)
    return [tuple(t) for t in tree]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double marker_radius_hanchuan(float x, float y, float z, cnp.ndarray[cnp.float32_t, ndim=3] img, double thr):
    cdef long long sz0 = img.shape[0], sz1 = img.shape[1], sz2 = img.shape[2], i, j, k
    cdef double max_r = sz0 / 2, total_num, background_num, ir, dz, dy, dx, zlower, zupper, r
    max_r = min(max_r, sz1 / 2)
    max_r = min(max_r, sz2 / 2)

    for ir in range(1, <int>(max_r + 1)):
        total_num = background_num = 0

        zlower = -ir
        zupper = ir
        for dz in range(<int>zlower, <int>(zupper + 1)):
            for dy in range(<int>-ir, <int>(ir + 1)):
                for dx in range(<int>-ir, <int>(ir + 1)):
                    total_num += 1
                    r = sqrt(dx*dx + dy*dy + dz*dz)
                    if ir - 1 < r <= ir:
                        i = <long long>(x + dx)
                        if i < 0 or i >= sz0:
                            return ir
                        j = <long long>(y + dy)
                        if j < 0 or j >= sz0:
                            return ir
                        k = <long long>(z + dz)
                        if k < 0 or k >= sz0:
                            return ir
                        if img[k, j, i] <= thr:
                            background_num += 1
                            if background_num > 0.001:
                                return ir
    return ir


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double marker_radius_hanchuan_xy(float x, float y, float z, cnp.ndarray[cnp.float32_t, ndim=3] img, double thr):
    cdef long long sz0 = img.shape[0], sz1 = img.shape[1], i, j
    cdef double max_r = sz0 / 2, total_num, background_num, ir, dy, dx, r
    max_r = min(max_r, sz1 / 2)

    for ir in range(1, <int>(max_r + 1)):
        total_num = background_num = 0

        for dy in range(<int>-ir, <int>(ir + 1)):
            for dx in range(<int>-ir, <int>(ir + 1)):
                total_num += 1
                r = sqrt(dx*dx + dy*dy)
                if ir - 1 < r <= ir:
                    i = <long long>(x + dx)
                    if i < 0 or i >= sz0:
                        return ir
                    j = <long long>(y + dy)
                    if j < 0 or j >= sz0:
                        return ir
                    if img[z, j, i] <= thr:
                        background_num += 1
                        if background_num > 0.001:
                            return ir
    return ir