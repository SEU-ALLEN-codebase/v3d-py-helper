cimport numpy as np
import numpy as np
from libc.math cimport sqrt, exp, pow
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef neuron_meanshift(tree: list[tuple], img: np.ndarray, window_radius=32, order=2.):
    """
    profile swc radius based on an image.

    :param tree: list of swc nodes 
    :param img: 3D neuronal image
    :param window_radius: the max radius of meanshift window
    :param order: the order to scale the intensity
    :return: 
    """
    assert img.ndim == 3
    tree_ = [list(t) for t in tree]
    cdef float x, y, z, r
    cdef np.ndarray[np.float32_t, ndim=3] img_ = img.astype(np.float32)
    for t in tree_:
        x, y, z = t[2:5]
        marker_meanshift(x, y, z, img_, window_radius, order)
        t[2:5] = x, y, z
    return [tuple(t) for t in tree]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef profile_radius(tree: list[tuple], img: np.ndarray, is2d: bool, bkg_thr: float):
    """
    profile swc radius based on an image.
    
    :param tree: list of swc nodes 
    :param img: 3D neuronal image
    :param is2d: 2D radius profiling
    :param bkg_thr: background threshold
    :return: 
    """
    assert img.ndim == 3
    tree_ = [list(t) for t in tree]
    cdef float x, y, z, r
    cdef np.ndarray[np.float32_t, ndim=3] img_
    if img.dtype == np.float32:
        img_ = img
    else:
        img_ = img.astype(np.float32)
    for t in tree_:
        x, y, z = t[2:5]
        if is2d:
            t[5] = marker_radius_hanchuan_xy(x, y, z, img_, bkg_thr)
        else:
            t[5] = marker_radius_hanchuan(x, y, z, img_, bkg_thr)
    return [tuple(t) for t in tree]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double marker_radius_hanchuan(float x, float y, float z, np.ndarray[np.float32_t, ndim=3] img, double thr):
    cdef long long sz0 = img.shape[2], sz1 = img.shape[1], sz2 = img.shape[0], i, j, k
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
                        if j < 0 or j >= sz1:
                            return ir
                        k = <long long>(z + dz)
                        if k < 0 or k >= sz2:
                            return ir
                        if img[k, j, i] <= thr:
                            background_num += 1
                            if background_num / total_num > 0.001:
                                return ir
    return ir


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef double marker_radius_hanchuan_xy(float x, float y, float z, np.ndarray[np.float32_t, ndim=3] img, double thr):
    cdef long long sz0 = img.shape[2], sz1 = img.shape[1], i, j, k = <long long>z
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
                    if j < 0 or j >= sz1:
                        return ir
                    if img[k, j, i] <= thr:
                        background_num += 1
                        if background_num / total_num > 0.001:
                            return ir
    return ir


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void marker_meanshift(float & x, float & y, float & z, np.ndarray[np.float32_t, ndim=3] img, int window_radius, double order):
    cdef:
        int wr_t = window_radius
        long long X, Y, Z, dX, dY, dZ
        double mean = 0., count = 0., w, center_dist, s2, prefactor, tot_x, tot_y, tot_z, sum_v, s, x_, y_, z_, dist
    for X in range( <int>max(x+.5-wr_t, 0), <int>min(img.shape[2] - 1, x + .5 + wr_t) ):
        for Y in range(<int> max(x + .5 - wr_t, 0), <int> min(img.shape[1] - 1, x + .5 + wr_t)):
            for Z in range(<int> max(x + .5 - wr_t, 0), <int> min(img.shape[0] - 1, x + .5 + wr_t)):
                dX = <int>(X - x)
                dY = <int>(Y - y)
                dZ = <int>(Z - z)
                dist = sqrt(dX*dX + dY*dY + dZ*dZ)
                if dist > wr_t:
                    continue
                w = pow(img[Z, Y, X], order)
                mean += w * dist
                count += w
    mean /= count + 1e-5
    wr_t = <int>max(1., mean)
    s = wr_t / 4.
    s2 = pow(s, 2)
    prefactor = 1. / pow(2 * np.pi * s2, 1.5)
    while True:
        tot_x = tot_y = tot_z = sum_v = 0.
        for X in range(<int> max(x + .5 - wr_t, 0), <int> min(img.shape[2] - 1, x + .5 + wr_t)):
            for Y in range(<int> max(x + .5 - wr_t, 0), <int> min(img.shape[1] - 1, x + .5 + wr_t)):
                for Z in range(<int> max(x + .5 - wr_t, 0), <int> min(img.shape[0] - 1, x + .5 + wr_t)):
                    dX = <int>(X - x)
                    dY = <int>(Y - y)
                    dZ = <int>(Z - z)
                    dist = sqrt(dX * dX + dY * dY + dZ * dZ)
                    if dist > wr_t:
                        continue
                    w = pow(img[Z, Y, X], order) * prefactor * exp(-dist / (2 * s2))
                    tot_x += w * X
                    tot_y += w * Y
                    tot_z += w * Z
                    sum_v += w
        if sum_v < 1e-5:
            break
        x_ = tot_x / sum_v
        y_ = tot_y / sum_v
        z_ = tot_z / sum_v
        center_dist = sqrt( (x-x_)*(x-x_) + (y-y_)*(y-y_) + (z-z_)*(z-z_) )
        x = x_
        y = y_
        z = z_
        wr_t -= <int>s
        if center_dist < 1:
            break


# cpdef neuron_radius_terafly(tree: list[tuple], img_path: str, is2d: bool, block_size: int, meanshift: bool, bkg_thr: float)