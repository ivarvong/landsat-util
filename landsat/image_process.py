# Pansharpened Image Process using Rasterio
# USGS Landsat Imagery Util
#
#
# Author: developmentseed
# Contributer: scisco, KAPPS-, kamicut
#
# License: CC0 1.0 Universal

import cPickle
import resource
import time
import warnings
import sys
from os.path import join, dirname
import tarfile
import numpy
import rasterio
import shutil
from rasterio.warp import reproject, RESAMPLING, transform

from skimage import img_as_ubyte, exposure
from skimage import transform as sktransform

import settings
from general_helper import Verbosity


class Process(Verbosity):
    """
    Image procssing class
    """

    def __init__(self, scene, bands=[4, 3, 2], src_path=None, dst_path=None, zipped=None, verbose=False):
        """
        @params
        scene - the scene ID
        bands - The band sequence for the final image. Must be a python list
        src_path - The path to the source image bundle
        dst_path - The destination path
        zipped - Set to true if the scene is in zip format and requires unzipping
        verbose - Whether to show verbose output
        """

        self.projection = {'init': 'epsg:3857'}
        self.dst_crs = {'init': u'epsg:3857'}
        self.scene = scene
        self.bands = bands
        self.pixel = 30
        self.src_path = src_path if src_path else dirname(dirname(__file__))
        self.dst_path = dst_path if dst_path else settings.PROCESSED_IMAGE

        self.output_file = join(self.dst_path, 'landsat-pan.TIF')
        self.verbose = verbose

        self.scene_path = join(self.src_path, scene)

        self.bands_path = []
        for band in self.bands:
            self.bands_path.append(join(self.scene_path, '%s_B%s.TIF' % (self.scene, band)))

        if zipped:
            self._unzip(join(self.src_path, self.scene) + '.tar.bz', join(self.src_path, self.scene), self.scene)

    def run(self, pansharpen=True):

        self.output("* Image processing started", normal=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rasterio.drivers():
                bands = []

                # Add bands 8 for pansharpenning
                if pansharpen:
                    self.bands.append(8)
                    self.pixel = 15

                bands_path = []
                for band in self.bands:
                    bands_path.append(join(self.scene_path, '%s_B%s.TIF' % (self.scene, band)))

                for i, band in enumerate(self.bands):
                    bands.append(self._read_band(bands_path[i]))

                src = rasterio.open(bands_path[-1])
                src_transform = src.transform
                src_shape = src.shape
                src_affine = src.affine
                src_crs = src.crs
                del src

                crn = self._get_bounderies(src_affine, src_shape, src_crs)

                dst_shape = (int((crn['lr']['x'][1][0] - crn['ul']['x'][1][0])/self.pixel),
                             int((crn['lr']['y'][1][0] - crn['ul']['y'][1][0])/self.pixel))

                dst_transform = (crn['ul']['x'][1][0], self.pixel, 0.0, crn['ul']['y'][1][0], 0.0, -self.pixel)

                del crn

                r = numpy.empty(dst_shape, dtype=numpy.uint16)
                g = numpy.empty(dst_shape, dtype=numpy.uint16)
                b = numpy.empty(dst_shape, dtype=numpy.uint16)
                b8 = numpy.empty(dst_shape, dtype=numpy.uint16)

                if pansharpen:
                    bands[:3] = self._rescale(bands[:3])

                new_bands = [r, g, b, b8]

                self.output("Projecting", normal=True, arrow=True)
                for i, band in enumerate(bands):
                    self.output("Projecting band %s" % (i + 1), normal=True, color='green', indent=1)
                    reproject(band, new_bands[i], src_transform=src_transform, src_crs=src_crs,
                              dst_transform=dst_transform, dst_crs=self.dst_crs, resampling=RESAMPLING.nearest)

                    f = open('band_%s' % i, 'w')
                    cPickle.dump(new_bands[i], f)
                    f.close()
                    new_bands[i] = None

                del new_bands
                del bands

                if pansharpen:
                    self.output("Pansharpening", normal=True, arrow=True)
                    # Pan sharpening
                    m = r + b + g
                    m = m + 0.1

                    self.output("calculating pan ratio", normal=True, color='green', indent=1)
                    pan = 1/m * b8
                    self.output("computing bands", normal=True, color='green', indent=1)

                    r = r * pan
                    b = b * pan
                    g = g * pan

                output = rasterio.open(self.output_file, 'w', driver='GTiff',
                                       width=dst_shape[1], height=dst_shape[0],
                                       count=3, dtype=numpy.uint8,
                                       nodata=0, transform=dst_transform, photometric='RGB',
                                       crs=self.dst_crs)

                for i in range(0, 3):
                    self.output("Color correcting band %s" % (i + 1), normal=True, arrow=True)

                    f = open('band_%s' % i, 'r')
                    obj = cPickle.load(f)
                    obj = obj.astype(numpy.uint16)

                    p2, p98 = self._percent_cut(obj)

                    obj = exposure.rescale_intensity(obj, in_range=(p2, p98))

                    if i == 0:
                        obj = exposure.adjust_gamma(obj, 1.1)

                    if i == 1:
                        obj = exposure.adjust_gamma(obj, 0.9)

                    self.output("Writing output band %s" % (i + 1), normal=True, arrow=True)
                    output.write_band(i+1, img_as_ubyte(obj))

                    del obj
                    shutil.rmtree('band_%s' % i)

                return self.output_file

    def _percent_cut(self, color):
        return numpy.percentile(color[numpy.logical_and(color > 0, color < 65535)], (2, 98))

    def _unzip(self, src, dst, scene):
        """ Unzip tar files """
        self.output("Unzipping %s - It might take some time" % scene, normal=True, arrow=True)
        tar = tarfile.open(src)
        tar.extractall(path=dst)
        tar.close()

    def _read_band(self, band_path):
        """ Reads a band with rasterio """
        return rasterio.open(band_path).read_band(1)

    def _rescale(self, bands):
        """ Rescale bands """
        self.output("Rescaling", normal=True, arrow=True)

        for key, band in enumerate(bands):
            self.output("Rescaling band %s" % (key + 1), normal=True, color='green', indent=1)
            bands[key] = sktransform.rescale(band, 2)
            bands[key] = (bands[key] * 65535).astype('uint16')

        return bands

    def _get_bounderies(self, src_affine, src_shape, src_crs):

        self.output("Getting bounderies", normal=True, arrow=True)
        output = {'ul': {'x': [0, 0], 'y': [0, 0]},  # ul: upper left
                  'lr': {'x': [0, 0], 'y': [0, 0]}}  # lr: lower right

        output['ul']['x'][0] = src_affine[2]
        output['ul']['y'][0] = src_affine[5]
        output['ul']['x'][1], output['ul']['y'][1] = transform(src_crs, self.projection,
                                                               [output['ul']['x'][0]],
                                                               [output['ul']['y'][0]])
        output['lr']['x'][0] = output['ul']['x'][0] + self.pixel * src_shape[0]
        output['lr']['y'][0] = output['ul']['y'][0] + self.pixel * src_shape[1]
        output['lr']['x'][1], output['lr']['y'][1] = transform(src_crs, self.projection,
                                                               [output['lr']['x'][0]],
                                                               [output['lr']['y'][0]])

        return output


class timer(object):
    """ A time class """
    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        self.end = time.time()
        print 'Time spent : {0:.2f} seconds'.format((self.end - self.start))


if __name__ == '__main__':

    rsrc = resource.RLIMIT_DATA
    resource.setrlimit(rsrc, (268435456, 268435456))
    soft, hard = resource.getrlimit(rsrc)
    print 'Soft limit starts as  :%s , hard: %s' % (soft, hard)

    with timer():
        p = Process(sys.argv[1],
                    src_path=sys.argv[2],
                    dst_path=sys.argv[2])


        print p.run(sys.argv[3] == 't')

