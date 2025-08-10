#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# tools for filtering the ionoephere estimates
################################################
#       Author: Yuan-Kai Liu, 2025
#       Caltech Seismological Laboratory
#       Contact: ykliu@caltech.edu
#       License: For academic use only
# ##############################################
#

import logging
logging.getLogger('h5py').setLevel(logging.WARNING)
import h5py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module='osgeo.gdal')

import os
from osgeo import gdal

import numpy as np
from skimage.transform import resize
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt

import isceobj
from isce.applications.gdal2isce_xml import gdal2isce_xml
from isceobj.Alos2Proc.Alos2ProcPublic import waterBodyRadar
from isceobj.TopsProc.runIon import adaptive_gaussian, weight_fitting

# label for waterBody file assigned int 8-bit convention
# LAND_LABEL    = 0
# WATER_LABEL   = -1
# NO_DATA_LABEL = -2

def read_h5_dataset(filename, datasetName=None):
    """
    Reads a specified dataset from an HDF5 file.
    If 'datasetName' is None, it reads the first dataset at root
    """
    with h5py.File(filename, 'r') as f:
        if datasetName is None:
            for key in f.keys():
                if isinstance(f[key], h5py.Dataset):
                    datasetName = key
                    break

        print(f'read datasetName: {datasetName}')
        dataset_obj = f[datasetName]
        data = dataset_obj[()]
        atr = dataset_obj.attrs
        return data, atr


def read_isce_band(filename, band=1):
    """Reads a specific band from an ISCE-compatible raster file.
    """
    #gdal.UseExceptions()
    filename  = str(filename)
    ds        = gdal.Open(filename, gdal.GA_ReadOnly)
    arr       = ds.GetRasterBand(band).ReadAsArray()
    no_data   = ds.GetRasterBand(band).GetNoDataValue()
    dtcode    = ds.GetRasterBand(band).DataType
    dtype_str = gdal.GetDataTypeName(dtcode).lower()

    if dtype_str == 'byte':
        dtype = np.uint8
    elif dtype_str == 'int16':
        dtype = np.int16
    elif dtype_str == 'int32':
        dtype = np.int32
    else:
        dtype = np.int16

    # get img dimensions
    width  = ds.RasterXSize
    height = ds.RasterYSize

    ds = None # close dataset
    return arr, (height, width), dtype, dtype_str, no_data


def create_isce_xml_header(filename, width, length, isce_dtype, bands=1,
                           nodata_value=None, extra_filename=None,
                           template_xml_path=None):
    """
    Creates/updates the ISCE XML header for a given file.
    All properties are inherited from template_xml_path if provided,
    then explicitly overridden by function arguments.
    """
    img             = isceobj.createImage()
    if template_xml_path:
        img.load(template_xml_path)

    img.filename    = filename
    img.width       = width
    img.length      = length
    img.dataType    = isce_dtype
    img.bands       = bands

    if nodata_value is not None:
        img.noDataValue = nodata_value
    if extra_filename:
        img.extraFilename = extra_filename

    print(f"isce2 xml header generated: {filename}.xml")

    img.renderHdr()


def write_isce2_file(datasets, outfile, ref_file):
    """
    Save datasets into different bands of an ISCE-compatible file.
    Assumes all input datasets have the same height and width.
    """
    #gdal.UseExceptions()
    height, width   = datasets[0].shape
    num_bands       = len(datasets)
    output_dtype    = datasets[0].dtype

    out             = np.zeros((height * num_bands, width), dtype=output_dtype)

    for i in range(num_bands):
        out[i : height * num_bands : num_bands, :] = datasets[i]

    out.tofile(outfile)

    isce_dtype_str  = ''
    if output_dtype == np.float32:
        isce_dtype_str = 'FLOAT'
    elif output_dtype == np.complex64:
        isce_dtype_str = 'CFLOAT'
    elif output_dtype == np.int8:    # For signed 8-bit integer
        isce_dtype_str = 'BYTE'      # ISCE typically uses 'BYTE' for 8-bit signed/unsigned
    elif output_dtype == np.uint8:   # For unsigned 8-bit integer
        isce_dtype_str = 'BYTE'
    elif output_dtype == np.int16:   # For signed 16-bit integer
        isce_dtype_str = 'SHORT'
    elif output_dtype == np.uint16:  # For unsigned 16-bit integer
        isce_dtype_str = 'USHORT'
    elif output_dtype == np.int32:   # For signed 32-bit integer
        isce_dtype_str = 'INT'
    elif output_dtype == np.uint32:  # For unsigned 32-bit integer
        isce_dtype_str = 'UINT'
    elif output_dtype == np.float64: # For 64-bit float (double precision)
        isce_dtype_str = 'DOUBLE'
    elif output_dtype == np.complex128: # For 128-bit complex (double precision complex)
        isce_dtype_str = 'CDOUBLE'
    else:
        # Fallback for unexpected types, or raise an error
        raise ValueError(f"Unsupported NumPy dtype for ISCE: {output_dtype}. "
                         f"Please add a mapping in write_isce2_file().")

    create_isce_xml_header(
        filename          = outfile,
        width             = width,
        length            = height,
        isce_dtype        = isce_dtype_str,
        bands             = num_bands,
        extra_filename    = outfile + '.vrt',
        template_xml_path = ref_file + '.xml'
    )



def find_largest_component(mask):
    """
    Finds the largest connected component where distinct non-zero labels (1, 2, 3, etc.)
    are treated as separate regions. Pixels that are 0, None, or False in the input
    mask are always treated as background.

    Args:
        mask (np.ndarray): The input mask array. Can be numeric (int/float labels),
                           boolean (False/True), or object (can contain None).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Boolean mask of the largest connected component. True only
                          for pixels that were originally non-zero, not None, not False.
            - np.ndarray: The full labeled array generated by skimage.measure.label,
                          reflecting components of distinct original labels.
    """

    # 1. Prepare a `processed_mask` for labeling.
    #    Pixels that are 0, False, or None in the original `mask` must be 0 in `processed_mask`.
    processed_mask = np.zeros_like(mask, dtype=mask.dtype) # Use original dtype for values

    # Boolean mask of pixels that are *valid foreground candidates* (not 0, not False, not None).
    is_valid_foreground_candidate = (mask != 0) # Handles numeric 0 and boolean False
    if mask.dtype == object:
        is_valid_foreground_candidate &= (mask != None)
    elif np.issubdtype(mask.dtype, np.floating):
        is_valid_foreground_candidate &= (~np.isnan(mask))

    # Populate `processed_mask`: only copy values from `mask` if they are valid foreground candidates.
    # This ensures `label` treats 0s, False, Nones, NaNs as background.
    processed_mask[is_valid_foreground_candidate] = mask[is_valid_foreground_candidate]


    # 2. Label connected components. `label` will find components for each distinct non-zero value.
    labeled = label(processed_mask)

    # 3. Handle edge case: no components found at all.
    if not labeled.max():
        return np.zeros_like(mask, dtype=bool), labeled

    # 4. Find the largest region among ALL identified components.
    #    `regionprops` considers all distinct components from `labeled`.
    regions = regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)

    # 5. Create a boolean mask specifically for the largest component.
    largest_component_mask = (labeled == largest.label)

    # 6. Final result: Ensure the returned mask only includes pixels that were part of
    #    the largest component AND were originally valid foreground pixels.
    final_largest_mask = largest_component_mask & is_valid_foreground_candidate

    return final_largest_mask, labeled



def fill_with_smoothed(data):
    """Replace the value of nan 'data' cells
    by the value of the linear interpolated data cell.
    The values, not covered by interpolation, are filled
    with nearest values.

    From isce3 code:
    https://github.com/isce-framework/isce3/blob/develop/python/packages/isce3/atmosphere/ionosphere_filter.py

    Parameters
    ----------
    data : numpy.ndarray
        array containing holes to be filled.
        nan values are considered as holes.

    Returns
    -------
    numpy.ndarray
        array with no data values filled with data values
        from numpy.griddata
    """
    rows, cols = data.shape
    x = np.arange(0, cols)
    y = np.arange(0, rows)
    xx, yy = np.meshgrid(x, y)

    xx = xx.ravel()
    yy = yy.ravel()
    data = data.ravel()

    is_nan_mask = np.isnan(data)
    not_nan_mask = np.invert(is_nan_mask)

    if np.all(not_nan_mask):
        return data.reshape([rows, cols])

    # find x and y where valid values are located.
    xx_wo_nan = xx[not_nan_mask]
    yy_wo_nan = yy[not_nan_mask]
    data_wo_nan = data[not_nan_mask]

    xnew = xx[np.isnan(data)]
    ynew = yy[np.isnan(data)]

    # linear interpolation with griddata
    znew = griddata((xx_wo_nan, yy_wo_nan),
                    data_wo_nan,
                    (xnew, ynew),
                    method='linear')
    data_filt = data.copy()
    data_filt[np.isnan(data)] = znew
    n_nonzero = np.sum(np.count_nonzero(np.isnan(data_filt)))

    if n_nonzero > 0:
        idx2= np.isnan(data_filt)

        xx_wo_nan = xx[np.invert(idx2)]
        yy_wo_nan = yy[np.invert(idx2)]
        data_wo_nan = data_filt[np.invert(idx2)]
        xnew = xx[idx2]
        ynew = yy[idx2]

        # extrapolation using nearest values
        znew_ext = griddata((xx_wo_nan, yy_wo_nan),
            data_wo_nan, (xnew, ynew), method='nearest')
        data_filt[np.isnan(data_filt)] = znew_ext
    return data_filt.reshape([rows, cols])


def fill_nearest(data, invalid=None):
    """Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell

    From isce3 code:
    https://github.com/isce-framework/isce3/blob/develop/python/packages/isce3/atmosphere/ionosphere_filter.py

    Parameters
    ----------
    data : numpy.ndarray
        array containing holes to be filled.
    invalid:
        a binary array of same shape as 'data'.
        data value are replaced where invalid is True
        If None (default), use: invalid  = np.isnan(data)

    Returns
    -------
    data[tuple(ind)]: numpy.ndarray
        array with no data values filled with data values
        from nearest neighborhood
    """
    if invalid is None:
        invalid = np.isnan(data)

    ind = distance_transform_edt(invalid,
                                return_distances=False,
                                return_indices=True)
    return data[tuple(ind)]



def multilook_and_save(infile, dst_width, dst_height, outfile=None, method='nearest'):
    """
    Resizes an ISCE raster mask to target dimensions, preserving labels and NoData.
    Creates/updates corresponding ISCE XML header.
    """
    order_map = {'nearest': 0, 'bilinear': 1, 'bicubic': 3}

    if outfile is not None and os.path.exists(outfile):
        print(f'Read from existing multilooked file: {outfile}.')
        out = read_isce_band(outfile)[0]

    else:
        print(f'Start multilooking file...')
        arr, arr_shape, dtype, dtype_str, no_data = read_isce_band(infile)
        target_shape = (dst_height, dst_width)
        out = resize(arr, target_shape, order=order_map[method], preserve_range=True, anti_aliasing=False)
        out = out.astype(dtype)

        if outfile is not None:
            print(f'Save multilooked file: {outfile}')
            write_isce2_file(datasets=[out], outfile=outfile, ref_file=infile)
            print(f'Resized mask saved: {outfile} ({dst_height} lines, {dst_width} samples)')
    return out



def make_radar_wbd(geom_basedir, input_wbd_file, ftype='Body'):
    """Creates a water body file in SAR radar coordinates."""
    radar_wbd_output_path = os.path.join(geom_basedir, f'water{ftype}.rdr')

    if os.path.exists(radar_wbd_output_path):
        print(f'Radar water body file already exists: {radar_wbd_output_path}. Skipping generation.')
    else:
        print(f'Generating radar water body file: {radar_wbd_output_path}')

        lat_file = os.path.join(geom_basedir, 'lat.rdr')
        lon_file = os.path.join(geom_basedir, 'lon.rdr')

        # Ensure XMLs exist for lat/lon for waterBodyRadar
        if not os.path.exists(lat_file + '.xml'): gdal2isce_xml(lat_file + '.vrt')
        if not os.path.exists(lon_file + '.xml'): gdal2isce_xml(lon_file + '.vrt')

        waterBodyRadar(lat_file, lon_file, input_wbd_file, radar_wbd_output_path)
        print(f'Successfully created radar water body file: {radar_wbd_output_path}')

        # fixImageXml.py is an external script, run it via os.system
        os.system(f'fixImageXml.py -i {radar_wbd_output_path} -f')
    return radar_wbd_output_path



def filtIon(ion, cor, size_min=100, size_max=200, corThresholdIon=0.85, pp=14, fit=False):
    """ exactly Cunren's adaptive gaussian filter
    From isce2 code
    https://github.com/isce-framework/isce2/blob/a492b8d76fc91fa82a100458b1714120b0fae090/contrib/stack/topsStack/filtIon.py
    """
    length, width = ion.shape

    ion_fit = weight_fitting(ion, cor, width, length, 1, 1, 1, 1, 2, corThresholdIon)

    #no fitting
    if fit == False:
        ion_fit *= 0

    ion -= ion_fit * (ion != 0)

    filt = adaptive_gaussian(ion, cor**pp, size_max, size_min)

    # compute image noise


    filt += ion_fit * (filt != 0)

    return filt
