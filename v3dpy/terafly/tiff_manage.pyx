from libc.stdint cimport uint32_t, uint16_t, uint8_t
from libc.string cimport memcpy
from libcpp.string cimport string
from libc.math cimport floor, ceil
import cython
cimport numpy as cnp
import numpy as np


cdef extern from "tiffio.h":
    ctypedef struct TIFF
    ctypedef uint32_t ttile_t
    ctypedef uint32_t tstrip_t
    ctypedef uint32_t tsize_t
    ctypedef void * tdata_t

    TIFF * TIFFOpen(const char * name, const char * mode)
    void TIFFClose(TIFF * tif)

    void TIFFSetWarningHandler(void * handler)
    void TIFFSetErrorHandler(void * handler)

    int TIFFGetField(TIFF * tif, unsigned int tag, ...)
    int TIFFReadDirectory(TIFF * tif)
    int TIFFIsTiled(TIFF * tif)
    tsize_t TIFFTileSize(TIFF * tif)
    ttile_t TIFFComputeTile(TIFF * tif, uint32_t x, uint32_t y, uint32_t z, uint32_t s)
    tsize_t TIFFReadEncodedTile(TIFF * tif, ttile_t tile, tdata_t buf, tsize_t size)
    tstrip_t TIFFNumberOfStrips(TIFF * tif)
    tsize_t TIFFStripSize(TIFF * tif)
    tsize_t TIFFReadEncodedStrip(TIFF * tif, tstrip_t strip, tdata_t buf, tsize_t size)
    tsize_t TIFFReadRawStrip(TIFF * tif, tstrip_t strip, tdata_t buf, tsize_t size)
    int TIFFIsByteSwapped(TIFF * tif)
    int TIFFSetDirectory(TIFF * tif, uint16_t dirnum)
    ttile_t TIFFNumberOfTiles(TIFF * tif)

    enum:  # TIFF tags
        TIFFTAG_IMAGEWIDTH
        TIFFTAG_IMAGELENGTH
        TIFFTAG_BITSPERSAMPLE
        TIFFTAG_SAMPLESPERPIXEL
        TIFFTAG_PAGENUMBER
        TIFFTAG_PHOTOMETRIC
        TIFFTAG_COMPRESSION
        TIFFTAG_PLANARCONFIG
        TIFFTAG_TILEWIDTH
        TIFFTAG_TILELENGTH
        TIFFTAG_TILEDEPTH
        TIFFTAG_ROWSPERSTRIP


cdef void swap2bytes(void* targetp):
    cdef unsigned char* tp = <unsigned char*>targetp
    cdef unsigned char a = tp[0]
    tp[0] = tp[1]
    tp[1] = a

cdef void swap4bytes(void* targetp):
    cdef unsigned char* tp = <unsigned char*>targetp
    cdef unsigned char a = tp[0]
    tp[0] = tp[3]
    tp[3] = a
    a = tp[1]
    tp[1] = tp[2]
    tp[2] = a


cdef void close_tiff3d_file(unsigned long long fhandle):
    TIFFClose(<TIFF*> fhandle)


cdef unsigned long long load_tiff3d2metadata(string filename, unsigned int& sz0, unsigned int& sz1, unsigned int& sz2,
                                            unsigned int& sz3, int& datatype, bint& b_swap):
    cdef:
        uint32_t XSIZE
        uint32_t YSIZE
        uint16_t bpp
        uint16_t spp
        uint16_t Cpage
        uint16_t Npages
        TIFF* input
        int check
        char* mode = 'r'

    # disable warning and error handlers to avoid messages on unrecognized tags
    TIFFSetWarningHandler(NULL)
    TIFFSetErrorHandler(NULL)

    input = TIFFOpen(filename.c_str(), mode)
    if not input:
        raise IOError("Cannot open the file.")

    check = TIFFGetField(input, TIFFTAG_IMAGEWIDTH, &XSIZE)
    if not check:
        TIFFClose(input)
        raise IOError("Image width of undefined.")

    check = TIFFGetField(input, TIFFTAG_IMAGELENGTH, &YSIZE)
    if not check:
        TIFFClose(input)
        raise IOError("Image length of undefined.")

    check = TIFFGetField(input, TIFFTAG_BITSPERSAMPLE, &bpp)
    if not check:
        TIFFClose(input)
        raise IOError("Undefined bits per sample.")

    check = TIFFGetField(input, TIFFTAG_SAMPLESPERPIXEL, &spp)
    if not check:
        spp = 1

    check = TIFFGetField(input, TIFFTAG_PAGENUMBER, &Cpage, &Npages)
    if check != 1 or Npages == 0:
        Npages = 0
        while TIFFReadDirectory(input):
            Npages += 1

    sz0 = XSIZE
    sz1 = YSIZE
    sz2 = Npages
    sz3 = spp
    datatype = bpp // 8

    b_swap = TIFFIsByteSwapped(input)
    return <unsigned long long> input


cdef void copydata(unsigned char *psrc, uint32_t stride_src, unsigned char *pdst, uint32_t stride_dst, uint32_t width, uint32_t len):
    cdef uint32_t i
    for i in range(len):
        memcpy(pdst, psrc, width * sizeof(unsigned char))
        psrc += stride_src
        pdst += stride_dst


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void read_tiff_3d_file_to_buffer(unsigned long long fhandler,
                                     unsigned char * img,
                                     unsigned int img_width,
                                     unsigned int img_height,
                                     unsigned int first,
                                     unsigned int last,
                                     bint b_swap,
                                     int downsampling_factor=1,
                                     int start_i=-1,
                                     int end_i=-1,
                                     int start_j=-1,
                                     int end_j=-1):
    cdef:
        uint32_t rps
        uint16_t spp, bpp, orientation, photo, comp, planar_config
        int check, strips_per_image, LastStripSize
        uint32_t XSIZE
        uint32_t YSIZE
        TIFF* input = <TIFF*>fhandler

    check = TIFFGetField(input, TIFFTAG_BITSPERSAMPLE, &bpp)
    if not check:
        raise IOError("Undefined bits per sample.")

    check = TIFFGetField(input, TIFFTAG_SAMPLESPERPIXEL, &spp)
    if not check:
        raise IOError("Undefined samples per sample.")

    check = TIFFGetField(input, TIFFTAG_PHOTOMETRIC, &photo)
    if not check:
        raise IOError("Cannot determine photometric interpretation.")

    check = TIFFGetField(input, TIFFTAG_COMPRESSION, &comp)
    if not check:
        raise IOError("Cannot determine compression technique.")

    check = TIFFGetField(input, TIFFTAG_PLANARCONFIG, &planar_config)
    if not check:
        raise IOError("Cannot determine planar configuration.")

    start_i = 0 if start_i == -1 else start_i
    end_i = img_height - 1 if end_i == -1 else end_i
    start_j = 0 if start_j == -1 else start_j
    end_j = img_width - 1 if end_j == -1 else end_j

    cdef:
        uint32_t tilewidth, tilelength, tiledepth
        tsize_t tilenum, tilesize, tilenum_width, tilenum_length
        ttile_t tile
        cnp.ndarray[cnp.uint8_t, ndim=1] npdata
        tdata_t data
        unsigned char *psrc  # pointer in the tile buffer to the top left pixel of the current block to be copied
        unsigned char *pdst  # pointer in the image buffer to the top left pixel of the current block to be filled
        uint32_t stride_src, stride_dst
        tsize_t i  # row index in the slice of the top left pixel of the current block to be filled
        tsize_t j  # column index in the slice of the top left pixel of the current block to be filled
        uint32_t width  # width of the current block to be filled (in pixels)
        uint32_t len  # length of the current block to be filled (in pixels)
        int page = 0

    check = TIFFIsTiled(input)
    if check:
        # checks
        if TIFFGetField(input, TIFFTAG_TILEDEPTH, &tiledepth):
            raise IOError("Tiling among slices (z direction) not supported.")
        if spp > 1:
            if TIFFGetField(input, TIFFTAG_PLANARCONFIG, &planar_config):
                if planar_config > 1:
                    raise IOError("Non-interleaved multiple channels not supported with tiling.")

        # tiling is in x,y only
        TIFFGetField(input, TIFFTAG_TILEWIDTH, &tilewidth)
        TIFFGetField(input, TIFFTAG_TILELENGTH, &tilelength)
        tilenum = TIFFNumberOfTiles(input)
        tilesize = TIFFTileSize(input)
        tilenum_width = (img_width % tilewidth) if (img_width / tilewidth) + 1 else img_width / tilewidth
        tilenum_length = (img_height % tilelength) if (img_height / tilelength) + 1 else img_height / tilelength

        npdata = np.zeros(tilesize, dtype=np.uint8)
        data = <unsigned char *> npdata.data
        stride_src = tilewidth * spp  # width of tile (in bytes)
        stride_dst = (end_j - start_j + 1) * spp  # width of subregion (in bytes)

        while True:
            # Calculate initial parameters
            psrc = <unsigned char *> data + ((start_i % tilelength) * tilewidth + (start_j % tilewidth)) * spp
            pdst = img
            len = tilelength - (start_i % tilelength)
            tile = TIFFComputeTile(input, start_j, start_i, 0, 0)

            # Loop over all tiles in the TIFF file
            i = start_i
            while i <= end_i:
                width = tilewidth - (start_j % tilewidth)
                j = start_j
                while j <= end_j:
                    # Read the current tile into the data buffer
                    TIFFReadEncodedTile(input, tile, data, <tsize_t> -1)

                    # Copy the current block from the tile buffer to the image buffer
                    copydata(psrc, stride_src, pdst, stride_dst, width * spp, len)

                    j += width
                    tile += 1
                    psrc = <unsigned char *> data + ((i % tilelength) * tilewidth) * spp
                    pdst += width * spp
                    width = tilewidth if ((tile % tilenum_width + 1) * tilewidth <= end_j + 1) else (end_j + 1) % tilewidth

                i += len
                tile = TIFFComputeTile(input, start_j, i, 0, 0)
                psrc = <unsigned char *> data + ((i % tilelength) * tilewidth + (start_j % tilewidth)) * spp
                pdst = img + (i - start_i) * stride_dst
                len = tilelength if ((tile / tilenum_width + 1) * tilelength <= end_i + 1) else (end_i + 1) % tilelength

                page += 1
            if not(page < (last - first + 1) and TIFFReadDirectory(input)):
                break
        return

    if not TIFFGetField(input, TIFFTAG_ROWSPERSTRIP, &rps):
        rps = img_height
    strips_per_image = (img_height + rps - 1) / rps
    last_strip_size = img_height % rps
    if last_strip_size == 0:
        last_strip_size = rps

    cdef unsigned char * buf = img
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] nprowbuf
    cdef unsigned char * rowbuf
    cdef unsigned char * bufptr
    cdef int strip_index

    if downsampling_factor == 1:  # read without downsampling
        if start_i < 0 or end_i >= img_height or start_j < 0 or end_j >= img_width or start_i >= end_i or start_j >= end_j:
            raise IOError("Wrong substack indices.")

        if start_i == 0 and end_i == (img_height - 1) and start_j == 0 and end_j == (img_width - 1):  # read whole images from files
            if not TIFFSetDirectory(input, first):
                raise IOError("Cannot open the requested first strip.")

            while True:
                for i in range(strips_per_image - 1):
                    if comp == 1:
                        TIFFReadRawStrip(input, i, buf, spp * rps * img_width * (bpp / 8))
                        buf += spp * rps * img_width * (bpp / 8)
                    else:
                        TIFFReadEncodedStrip(input, i, buf, spp * rps * img_width * (bpp / 8))
                        buf += spp * rps * img_width * (bpp / 8)

                if comp == 1:
                    TIFFReadRawStrip(input, strips_per_image - 1, buf, spp * last_strip_size * img_width * (bpp / 8))
                else:
                    TIFFReadEncodedStrip(input, strips_per_image - 1, buf, spp * last_strip_size * img_width * (bpp / 8))
                buf += spp * last_strip_size * img_width * (bpp / 8)

                page += 1
                if not (page < last - first + 1 and TIFFReadDirectory(input)):
                    break
        else:  # read only a subregion of images from files
            if not TIFFGetField(input, TIFFTAG_IMAGEWIDTH, &XSIZE):
                raise IOError("Image width of undefined.")
            if not TIFFGetField(input, TIFFTAG_IMAGELENGTH, &YSIZE):
                raise IOError("Image length of undefined.")

            nprowbuf = np.zeros(spp * rps * XSIZE * (bpp/8), dtype=np.uint8)
            rowbuf = <uint8_t *> nprowbuf.data

            while True:
                if not TIFFSetDirectory(input, first + page):
                    raise IOError("Cannot open next requested strip.")

                strip_index = (start_i / rps) - 1  # the strip preceeding the first one
                for i in range(start_i, end_i + 1):
                    if floor(i / rps) > strip_index:  # read a new strip
                        strip_index = int(floor(i / rps))
                        if comp == 1:
                            TIFFReadRawStrip(input, strip_index, rowbuf,
                                             spp * (rps if strip_index < strips_per_image else last_strip_size) * XSIZE * (bpp / 8))
                        else:
                            TIFFReadEncodedStrip(input, strip_index, rowbuf,
                                                 spp * (rps if strip_index < strips_per_image else last_strip_size) * XSIZE * (bpp / 8))

                    bufptr = rowbuf + <uint32_t>((i % rps) * (spp * XSIZE * (bpp / 8)))
                    if bpp == 8:
                        for j in range(end_j - start_j + 1):
                            for c in range(spp):
                                buf[j * spp + c] = bufptr[j * spp + c]
                    else:
                        for j in range(end_j - start_j + 1):
                            for c in range(spp):
                                (<uint16_t *> buf)[j * spp + c] = (<uint16_t *> bufptr)[j * spp + c]
                    buf += spp * (end_j - start_j + 1) * (bpp / 8)

                page += 1
                if not (page < last - first + 1):
                    break
    else:  # read with downsampling
        # preliminary checks
        if start_i != 0 or end_i != (img_height - 1) or start_j != 0 or end_j != (img_width - 1):  # a subregion has been requested
            raise IOError("Subregion extraction not supported with downsampling.")
        if not TIFFGetField(input, TIFFTAG_IMAGEWIDTH, &XSIZE):
            raise IOError("Image width of undefined.")
        if not TIFFGetField(input, TIFFTAG_IMAGELENGTH, &YSIZE):
            raise IOError("Image length of undefined.")

        if ceil(XSIZE / downsampling_factor) < img_width:
            raise IOError("Requested image width too large.")
        if ceil(YSIZE / downsampling_factor) < img_height:
            raise IOError("Requested image height too large.")

        nprowbuf = np.zeros(spp * rps * XSIZE * (bpp / 8), dtype=np.uint8)
        rowbuf = <uint8_t *> nprowbuf.data

        while True:
            if not TIFFSetDirectory(input, (first + page) * downsampling_factor):
                raise IOError("Cannot open next requested strip.")

            strip_index = -1  # the strip preceeding the first one
            for i in range(img_height):
                if floor(i * downsampling_factor / rps) > strip_index:  # read a new strip
                    strip_index = int(floor(i * downsampling_factor / rps))
                    if comp == 1:
                        TIFFReadRawStrip(input, strip_index, rowbuf,
                                         spp * (rps if strip_index < strips_per_image else last_strip_size) * XSIZE * (bpp / 8))
                    else:
                        TIFFReadEncodedStrip(input, strip_index, rowbuf,
                                             spp * (rps if strip_index < strips_per_image else last_strip_size) * XSIZE * (bpp / 8))

                bufptr = rowbuf + <uint32_t>(((i * downsampling_factor) % rps) * (spp * XSIZE * (bpp / 8)))
                if bpp == 8:
                    for j in range(img_width):
                        for c in range(spp):
                            buf[j * spp + c] = bufptr[j * spp * downsampling_factor + c]
                else:
                    for j in range(img_width):
                        for c in range(spp):
                            (<uint16_t *> buf)[j * spp + c] = (<uint16_t *> bufptr)[j * spp * downsampling_factor + c]
                buf += spp * img_width * (bpp / 8)

            page += 1

            if not (page < last - first + 1):
                break

    cdef tsize_t total = img_width * img_height * spp * (last-first+1)
    if b_swap:
        if bpp / 8 == 2:
            for i in range(total):
                swap2bytes(<void *> (img + 2 * i))
        elif bpp / 8 == 4:
            for i in range(total):
                swap4bytes(<void *> (img + 4 * i))