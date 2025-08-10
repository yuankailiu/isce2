#!/usr/bin/env python3

# Author: Cunren Liang
# Copyright 2021
#
# **************************************
# Filter ionospheric phase screen using
#       adaptive Gaussian filtering and
#       coherence/amp-based masking
#
#       Author: Yuan-Kai Liu 2025, Cunren Liang 2021
#       Caltech Seismological Laboratory
#       Contact: ykliu@caltech.edu
#       License: For academic use only
#
# Description:
#       This script applies connected-component
#       masking and adaptive Gaussian smoothing
#       to ionospheric phase delay maps.
# **************************************

#import copy
#import glob
#import shutil
import argparse
import os

import isce
import isceobj
import numpy as np
from filtIon_utils import (create_isce_xml_header, fill_nearest,
                           fill_with_smoothed, filtIon, find_largest_component,
                           make_radar_wbd, multilook_and_save, read_h5_dataset,
                           read_isce_band, write_isce2_file)
# Yuan-Kai Liu (2025): Future, we can combine filtIon_utils into TopsProc.runIon
#from isceobj.TopsProc.runIon import adaptive_gaussian
#from isceobj.TopsProc.runIon import weight_fitting
from skimage.transform import resize


def createParser():
    parser = argparse.ArgumentParser(description='filtering ionosphere')
    parser.add_argument('-i', '--input', dest='input', type=str, required=True,
            help='input ionosphere')
    parser.add_argument('-c', '--coherence', dest='coherence', type=str, required=True,
            help='coherence')
    parser.add_argument('-o', '--output', dest='output', type=str, required=True,
            help='output ionosphere')
    parser.add_argument('-a', '--win_min', dest='win_min', type=int, default=100,
            help='minimum filtering window size')
    parser.add_argument('-b', '--win_max', dest='win_max', type=int, default=200,
            help='maximum filtering window size')
    #parser.add_argument('-m', '--masked_areas', dest='masked_areas', type=int, nargs='+', action='append', default=None,
    #        help='This is a 2-d list. Each element in the 2-D list is a four-element list: [firstLine, lastLine, firstColumn, lastColumn], with line/column numbers starting with 1. If one of the four elements is specified with -1, the program will use firstLine/lastLine/firstColumn/lastColumn instead. e.g. two areas masked out: --masked_areas 10 20 10 20 --masked_areas 110 120 110 120')

    # *** Yuan-Kai Liu (2025) ***
    parser.add_argument('-m' , '--maskfile', dest='maskfile', type=str, default=None,
            help='Path to custom mask file.')
    parser.add_argument('-w', '--wbdfile', dest='wbdfile', type=str, default=None,
            help='Path to original water body file.')
    parser.add_argument('-hgt', '--height_max', dest='height_max', type=float, default=4500.,
            help='Elevation (maximum) cutoff. Default=%(default)s')
    parser.add_argument('--amp_cutoff', dest='amp_cutoff', type=float, default=0.0,
            help='cutoff for amplitude. float for percentage, int for actual amp value. Default=%(default)s')
    parser.add_argument('--cor_cutoff', dest='cor_cutoff', type=float, default=0.75,
            help='cutoff for coherence. Default=%(default)s')
    parser.add_argument('-it', '--iteration', dest='iteration', type=int, default=1,
            help='Number of iterations for filling-filtering. Default=%(default)s')
    parser.add_argument('-f', '--fill', dest='fill', type=str, default='zero',
            help='Fill masked data with either {zero, nearest, smooth}. Default=%(default)s')
    # ***************************

    return parser


def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def main(iargs=None):
    '''
    check overlap among all acquistions, only keep the bursts that in the common overlap,
    and then renumber the bursts.
    '''
    inps = cmdLineParse(iargs)

    '''
    This function filters image using gaussian filter

    we projected the ionosphere value onto the ionospheric layer, and the indexes are integers.
    this reduces the number of samples used in filtering
    a better method is to project the indexes onto the ionospheric layer. This way we have orginal
    number of samples used in filtering. but this requries more complicated operation in filtering
    currently not implemented.
    a less accurate method is to use ionsphere without any projection
    '''

    #################################################
    #SET PARAMETERS HERE
    #if applying polynomial fitting
    #False: no fitting, True: with fitting
    fit = True
    #gaussian filtering window size
    size_max = inps.win_max
    size_min = inps.win_min

    #THESE SHOULD BE GOOD ENOUGH, NO NEED TO SET IN setup(self)
    corThresholdIon = 0.85
    #################################################

    print('filtering ionosphere')
    #I find it's better to use ionosphere that is not projected, it's mostly slowlying changing anyway.
    #this should also be better for operational use.
    ionfile = inps.input
    #since I decide to use ionosphere that is not projected, I should also use coherence that is not projected.
    corfile = inps.coherence

    #use ionosphere and coherence that are projected.
    #ionfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionRaw)
    #corfile = os.path.join(ionParam.ionDirname, ionParam.ioncalDirname, ionParam.ionCor)

    outfile = inps.output

    # ************
    # parameters for msking
    maskfile    = inps.maskfile
    wbdfile     = inps.wbdfile
    cor_cutoff  = inps.cor_cutoff
    amp_cutoff  = inps.amp_cutoff
    hgt_max     = inps.height_max
    iteration   = inps.iteration
    out_dir     = os.path.dirname(outfile)
    maskout     = os.path.join(out_dir, 'filt_msk.rdr')
    fill        = inps.fill

    # paths
    base_dir    = ionfile.split('ion/')[0]
    pair12      = ionfile.split('ion/')[1].split('/')[0]
    geom_dir    = os.path.join(base_dir, 'merged', 'geom_reference')
    pair_dir    = os.path.join(base_dir, 'ion', pair12)
    # ************


    img = isceobj.createImage()
    img.load(ionfile + '.xml')
    width = img.width
    length = img.length
    ion = (np.fromfile(ionfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    cor = (np.fromfile(corfile, dtype=np.float32).reshape(length*2, width))[1:length*2:2, :]
    amp = (np.fromfile(ionfile, dtype=np.float32).reshape(length*2, width))[0:length*2:2, :]

    ########################################################################################
    #AFTER COHERENCE IS RESAMPLED AT grd2ion, THERE ARE SOME WIRED VALUES
    cor[np.nonzero(cor<0)] = 0.0
    cor[np.nonzero(cor>1)] = 0.0
    ########################################################################################


    ## 1. ***  user defined mask file  ***
    # need to be same dim as raw_no_projection.ion
    if maskfile is not None:
        if maskfile.endswith('.h5'):
                msk_use = read_h5_dataset(maskinfile)[0]
        else:
                msk_use = read_isce_band(maskinfile)[0]
                msk_use = msk_use==0   # assume isce2 mask assign land as 0
        if msk_use.shape != ion.shape:
            msk_use = resize(msk_use, (length,width), order=0, preserve_range=True, anti_aliasing=False)
        msk_use = msk_use.astype(bool)
    else:
        msk_use = None


    ## 2.1 ***  waterBody mask file  ***
    if wbdfile is not None:
        print("making waterBody in radar coordinates")
        wbdfile_rdr = make_radar_wbd(geom_dir, wbdfile)

        print("resampling waterBody to ionosphere product sizes")
        wbdfile_mlk = os.path.join(geom_dir, 'waterBody_ionlk.rdr')
        wbd_mlk = multilook_and_save(wbdfile_rdr, width, length, wbdfile_mlk, method='nearest')
        wbd_use = wbd_mlk==0  # get land
    else:
        wbd_use = None


    ## 2.2 ***  elevation file  ***
    hgtfile_rdr = os.path.join(geom_dir, 'hgt.rdr')
    if hgtfile_rdr is not None:
        print(f"resampling elevation file to ionosphere product sizes, height_max={hgt_max}")
        hgtfile_mlk = os.path.join(geom_dir, 'hgt_ionlk.rdr')
        hgt_mlk = multilook_and_save(hgtfile_rdr, width, length, hgtfile_mlk, method='nearest')
        hgt_use = hgt_mlk < hgt_max  # get below elevation threshold
        hgt_use = find_largest_component(hgt_use)[0]
    else:
        hgt_use = None


    ## 3. ***  subband conncomp masking  ***
    # lower & lower conncomp (requires further downlook to match the `ionfile`)
    lp_connFile = os.path.join(pair_dir, 'lower', 'merged', 'fine.unw.conncomp')
    hp_connFile = os.path.join(pair_dir, 'upper', 'merged', 'fine.unw.conncomp')
    if os.path.exists(lp_connFile) and os.path.exists(hp_connFile):
        print('use sub-band connComp for masking')
        lp_conn = multilook_and_save(lp_connFile, width, length)
        hp_conn = multilook_and_save(hp_connFile, width, length)

        # save the common largest region
        lp_conn = find_largest_component(lp_conn)[0]
        hp_conn = find_largest_component(hp_conn)[0]
        conn_use = lp_conn * hp_conn
        write_isce2_file(datasets=[conn_use.astype(np.int8)],
                         outfile=os.path.join(pair_dir, 'ion_cal','common.unw.conncomp'),
                         ref_file=lp_connFile)

    else:
        conn_use = None


    ## 4. ***  coherence masking  ***
    if cor_cutoff != 0.0:
        print(f'coherence cutoff value = {cor_cutoff}')
        cor_cutoff = np.max([cor_cutoff, 0.75])  # ensure conservative value
        cor_use = cor > cor_cutoff
        cor_use = find_largest_component(cor_use)[0]
    else:
        cor_use = None


    ## 5. ***  amplitude masking  ***
    if amp_cutoff != 0.0:
        if isinstance(amp_cutoff, float):
            print(f'use approximate {amp_cutoff}% amplitude distribution for masking')
            amp_cutoff = np.nanpercentile(amp, amp_cutoff*1e2)
        amp_cutoff = np.max([amp_cutoff, 101])  # ensure conservative value
        print(f'amplitude cutoff value = {amp_cutoff}')
        amp_use = (amp > amp_cutoff)
        amp_use = find_largest_component(amp_use)[0]
    else:
        amp_use = None


    ## ***  unify all masks  ***
    msk = (ion != 0) & (~np.isnan(ion)) & (cor != 0.0) & (~np.isnan(cor)) & (amp != 0.0) & (~np.isnan(amp))
    if msk_use  is not None: msk *= msk_use
    if wbd_use  is not None: msk *= wbd_use
    if hgt_use  is not None: msk *= hgt_use
    if conn_use is not None: msk *= conn_use
    if cor_use  is not None: msk *= cor_use
    if amp_use  is not None: msk *= amp_use


    # save with wbd mask format
    if wbd_use is not None:
        write_isce2_file(datasets=[msk.astype(np.int8)], outfile=maskout, ref_file=wbdfile_mlk)
        print(f"Ionophere filtering mask written to: {maskout}")


    ## ******  iterative masking/filling/masking  ******
    if iteration > 0:
        print(f'*** start iterative masking / filling ({fill}) / adaptive gaussian filtering ***')
        msk_keep = msk.copy()
        msk_drop = ~msk

        for iter_count in range(iteration):
            print(f'iteration {iter_count+1} of {iteration}')
            if iter_count == 0:
                ion_fin = ion.copy()

            ion_fin[msk_drop] = np.nan

            # fill
            if fill == 'nearest':
                print(fill)
                ion_fin = fill_nearest(ion_fin, invalid=msk_drop)
            elif fill == 'smooth':
                print(fill)
                ion_fin = fill_with_smoothed(ion_fin)
            else: # filled simply with zero
                print(fill)
                ion_fin[msk_drop]  = 0.0


            # replace in later iterations
            if iter_count > 0:
                ion_fin[msk_keep] = ion[msk_keep]

            # run adaptive filter
            ion_fin = filtIon(ion_fin, cor, size_min=size_min, size_max=size_max,
                               corThresholdIon=0.85, fit=fit)

        # output file
        out_img = write_isce2_file([amp, ion_fin], outfile, ionfile)
        print(f"Filtered ionosphere written to: {outfile}")
    ## *******************  done  **********************



if __name__ == '__main__':
    '''
    Main driver.
    '''
    # Main Driver
    main()

