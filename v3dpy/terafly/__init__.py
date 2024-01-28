import os

from .volume_managers import VirtualVolume, TiledVolume
from pathlib import Path
from .config import *


class TeraflyInterface:
    """
    Currently only support 3D tiff tiles.
    """
    def __init__(self, path: os.PathLike | str):
        self._path = Path(path)
        self._volume: VirtualVolume
        self.update_metadata()

    def update_metadata(self):
        """
        now it only supports TiledVolume (Tiled Tiff/V3DRaw 3D)
        :param path: teraconverted brain / resolution
        :return: an instance of the image manager from the path
        """
        self._volume = None
        if self._path.is_dir():
            format = "unknown"
            if (self._path / FORMAT_MDATA_FILE_NAME).is_file():
                try:
                    with open(self._path / FORMAT_MDATA_FILE_NAME) as f:
                        format = f.readline().rstrip()
                    if format == TILED_MC_FORMAT:
                        raise NotImplementedError
                    elif format == STACKED_FORMAT:
                        raise NotImplementedError
                    elif format == TILED_FORMAT:
                        self._volume = TiledVolume(self._path)
                    elif format == SIMPLE_FORMAT:
                        raise NotImplementedError
                    elif format == SIMPLE_RAW_FORMAT:
                        raise NotImplementedError
                    elif format == TIME_SERIES:
                        raise NotImplementedError
                    else:
                        raise ValueError(f"Cannot recognize format {format}")
                except:
                    raise f"Cannot import {format} at {self._path}"
            if self._volume is None:
                # try:
                #     self._volume = TiledMCVolume(path)
                # except:
                #     print(f"cannot import TiledMCVolume at {path}")
                #     try:
                #         self._volume = StackedVolume(path)
                #     except:
                #         print(f"cannot import StackedVolume at {path}")
                try:
                    self._volume = TiledVolume(self._path)
                except:
                    print(f"Cannot import TiledVolume at {self._path}")
                    # try:
                    #     self._volume = SimpleVolume(path)
                    # except:
                    #     print(f"cannot import SimpleVolume at {path}")
                    #     try:
                    #         self._volume = SimpleVolumeRaw(path)
                    #     except:
                    #         print(f"cannot import SimpleVolumeRaw at {path}")
                    #         try:
                    #             self._volume = TimeSeries(path)
                    #         except:
                    #             print(f"cannot import TimeSeries at {path}")
                    raise f"Generic error occurred when importing self._volume at {self._path}"
        elif self._path.is_file():
            if self._path.suffix in ['.raw', '.v3draw', '.RAW', '.V3DRAW', '.tif', '.tiff', '.TIF', '.TIFF']:
                raise NotImplementedError
            elif self._path.suffix in ['.xml', '.XML']:
                raise NotImplementedError
            else:
                raise ValueError(f"Unsupported file extensions for {self._path}")
        else:
            raise ValueError(f"Path {self._path} does not exist")

    def get_dim(self) -> tuple[int, int, int, int]:
        """
        Get TeraConvert dimensionality range.
        :return: (max x, max y, max z, max c)
        """
        return self._volume.DIM_H, self._volume.DIM_V, self._volume.DIM_D, self._volume.DIM_C

    def get_sub_volume(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int):
        """
        Different from Vaa3D, it returns the image of its original pixel type.
        The indexing is pixel-wise, so you have to use different coordinates for different resolutions
        If the loaded image is tif, it will be properly flipped

        :param path: teraconverted brain / resolution
        :param x0: starting x
        :param x1: ending x
        :param y0: starting y
        :param y1: ending y
        :param z0: starting z
        :param z1: ending z
        :return: the image crop
        """
        return self._volume.load_sub_volume(y0, y1, x0, x1, z0, z1)
