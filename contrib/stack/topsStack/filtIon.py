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
from filtIon_utils import (align_subswaths, create_isce_xml_header,
                           fill_nearest, fill_with_smoothed, filtIon,
                           find_largest_component, make_radar_wbd,
                           multilook_and_save, read_h5_dataset, read_isce_band,
                           write_isce2_file)
# Yuan-Kai Liu (2025): Future, we can combine filtIon_utils into TopsProc.runIon
#from isceobj.TopsProc.runIon import adaptive_gaussian
from isceobj.TopsProc.runIon import weight_fitting
from skimage.filters import threshold_multiotsu
from skimage.transform import resize
import matplotlib.pyplot as plt


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
    parser.add_argument('-hgt', '--height_max', dest='height_max', type=float, default=5000.,
            help='Elevation (maximum) cutoff. Default=%(default)s')
    parser.add_argument('--amp_cutoff', dest='amp_cutoff', type=float, default=0.0,
            help='cutoff for amplitude. float for percentage, int for actual amp value. Default=%(default)s')
    parser.add_argument('--cor_cutoff', dest='cor_cutoff', type=float, default=0.75,
            help='cutoff for coherence. Default=%(default)s')
    parser.add_argument('-it', '--iteration', dest='iteration', type=int, default=1,
            help='Number of iterations for filling-filtering. Default=%(default)s')
    parser.add_argument('-f', '--fill', dest='fill', type=str, default='zero',
            help='Fill masked data with either {zero, nearest, smooth}. Default=%(default)s')
    parser.add_argument('--swath_align', dest='swath_align', action='store_true', default=False,
            help='use masked-in values to offset sub-swath ions. Default=%(default)s')
    parser.add_argument('--multi_align', dest='multi_align', action='store_true', default=False,
            help='use masked-in values to offset multiple arbitrary ion chuncks. Default=%(default)s')
    # ***************************

    return parser


def cmdLineParse(iargs = None):
    parser = createParser()
    return parser.parse_args(args=iargs)


def drop_MAD_outliers(data, nsig=3):
    data = np.abs(data)
    MAD  = np.nanmedian(np.abs(data - np.nanmedian(data)))
    outliers = data > (nsig * MAD)
    print(f'drop outliers from {nsig}*MAD:', np.sum(outliers))
    return outliers


def otsu_score(data, thresholds):
    """Compute between-class variance score for given thresholds (vectorized)."""
    values = data[np.isfinite(data) & (data != 0)]
    bins = np.digitize(values, bins=thresholds)

    mu_T = values.mean()
    n = len(values)

    # total weight per bin
    counts = np.bincount(bins, minlength=bins.max()+1)
    # sum of values per bin
    sums = np.bincount(bins, weights=values, minlength=bins.max()+1)

    # avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        means = np.where(counts > 0, sums / counts, 0)
        weights = counts / n
        score = np.sum(weights * (means - mu_T)**2)
    return score


def adaptive_otsu(data, min_classes=3, max_classes=6):
    """Search best number of classes based on between-class variance."""
    values = data[np.isfinite(data) & (data != 0)]
    best_score, best_k, best_thresh = -np.inf, None, None

    for k in range(min_classes, max_classes+1):
        try:
            print(' try n_class=', k)
            thresholds = threshold_multiotsu(values, classes=k, nbins=128)
            score = otsu_score(values, thresholds)
            if score > best_score:
                best_score, best_k, best_thresh = score, k, thresholds
            print(f"  score={score:.3g}, best={best_score:.3g}, (k={best_k})")
        except Exception:
            continue

    print(f"[adaptive_otsu] Best split: {best_k} classes, score={best_score:.3g}")
    return best_thresh, best_k


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
                msk_use = read_h5_dataset(maskfile)[0]
        else:
                msk_use = read_isce_band(maskfile)[0]
                msk_use = msk_use==1
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
    msk = find_largest_component(msk)[0]
    msk_drop = np.array(~msk)

    # save this initial mask
    if wbd_use is not None:
        _maskout = maskout.replace('.rdr', '_init.rdr')
        write_isce2_file(datasets=[~msk_drop.astype(np.int8)], outfile=_maskout, ref_file=wbdfile_mlk)
        print(f"Ionophere filtering mask written to: {maskout}")


    # swath offsets alignment (update ion in-place)
    if inps.swath_align or inps.multi_align:
        ionfiles = [ionfile.replace('ion_cal', f'ion_cal_IW{i}') for i in (1, 2, 3)]
        # assume that all swaths exist
        haveIW = all(os.path.exists(f) for f in ionfiles)
        if inps.multi_align:
            haveIW = False
        # --- Build swath regions ---
        if haveIW:
            nclasses = 3
            print("[swath_align] Found IW-specific ion files, using them to define swaths.")
            regions = np.zeros_like(ion, dtype=int)
            for i, f in enumerate(ionfiles):
                ionIW = np.fromfile(f, dtype=np.float32).reshape(2*length, width)[1:length*2:2, :]
                regions[np.isfinite(ionIW) & (ionIW != 0.0)] = i
        else:
            print("[multi_align] Use adaptive Multi-Otsu.")
            thresholds, nclasses = adaptive_otsu(ion[ion != 0], min_classes=3, max_classes=6)
            regions = np.digitize(ion, bins=thresholds)

        # --- Compute offsets using masked version ---
        ion_unalign = ion.astype(float)
        ion_unalign[ion_unalign == 0.0] = np.nan
        ion_unalign[msk_drop] = np.nan

        _, offsets = align_subswaths(ion_unalign, regions, nswaths=nclasses)
        print(f"Applied swath offsets (rad): {offsets}")

        # --- Apply offsets to original full phase ---
        for i in range(nclasses):
            if offsets[i] != 0.0:
                ion[regions == i] -= offsets[i]

        ion = np.nan_to_num(ion, nan=0.0)

        # --- Save aligned ionosphere ---
        out_ionfile = ionfile.replace('raw_no_projection.ion', 'raw_no_projection_aligned.ion')
        out_img = write_isce2_file([amp, ion], out_ionfile, ionfile)
        print(f"Aligned ion file saved: {out_ionfile}")

        del ion_unalign

    raw = np.array(ion)

    ## ******  iterative masking/filling/masking  ******
    if iteration > 0:
        print(f'*** start iterative masking / filling ({fill}) / adaptive gaussian filtering ***')

        for iter_count in range(iteration):
            print(f'iteration {iter_count+1} of {iteration}')
            if iter_count == 0:
                ion_fin = np.array(ion)

            # drop outliers via polyfit (input mask set weight to zero)
            weight = cor.copy()
            weight[msk_drop] = 0.0
            ion_fit = weight_fitting(ion, weight, width, length, 1, 1, 1, 1, 2, corThresholdIon)
            outliers = drop_MAD_outliers(ion_fin - ion_fit, 4)

            if False: # debug plots
                def wrap_phase(x, period=2*np.pi, center=0.0):
                    """Wrap values into a range centered at `center` with given period."""
                    return (x - center + period/2) % period - period/2 + center
                # examples: wrap to -/+ 24
                wrapped_raw  = wrap_phase(raw, period=24.)
                wrapped_fin  = wrap_phase(ion_fin, period=24.)
                wrapped_fit  = wrap_phase(ion_fit, period=24.)
                wrapped_res  = wrap_phase(ion_fin - ion_fit, period=24.)
                nono = cor**10
                fig, axs = plt.subplots(ncols=6, figsize=[12,4])
                im=axs[0].imshow(nono * wrapped_raw, cmap='hsv'); plt.colorbar(im, ax=axs[0], label="wrapped rad"); axs[0].set_title("Raw wrapped")
                im=axs[1].imshow(nono * wrapped_fin, cmap='hsv'); plt.colorbar(im, ax=axs[1], label="wrapped rad"); axs[1].set_title("ion_fin wrapped")
                im=axs[2].imshow(nono * wrapped_fit, cmap='hsv'); plt.colorbar(im, ax=axs[2], label="wrapped rad"); axs[2].set_title("ion_fit wrapped")
                im=axs[3].imshow(nono * wrapped_res, cmap='hsv'); plt.colorbar(im, ax=axs[3], label="wrapped rad"); axs[3].set_title("Residual wrapped")
                axs[4].imshow(msk_drop); axs[4].set_title("Dropped")
                axs[5].imshow(msk_drop + outliers); axs[5].set_title("(and bad polyfits)")
                plt.show(block=False)
                plt.pause(0.1)   # brief pause to render

            if False: # don't do polyfit exclusion, killing pixels too harsh
                msk_drop = msk_drop + outliers

            # fill
            if fill == 'nearest':
                print(fill)
                ion_fin[msk_drop] = np.nan
                ion_fin = fill_nearest(ion_fin, invalid=msk_drop)
            elif fill == 'smooth':
                print(fill)
                ion_fin[msk_drop] = np.nan
                ion_fin = fill_with_smoothed(ion_fin)
            else: # filled simply with zero
                print(fill)
                ion_fin[msk_drop]  = 0.0


            # replace in later iterations
            if iter_count > 0:
                ion_fin[~msk_drop] = ion[~msk_drop].copy()

            # run adaptive filter
            ion_fin = filtIon(ion_fin, cor, size_min=size_min, size_max=size_max,
                               corThresholdIon=corThresholdIon, fit=fit)

            # drop outliers after filtering
            outliers = drop_MAD_outliers(ion_fin - ion)
            msk_drop = msk_drop + outliers

        # output file
        out_img = write_isce2_file([amp, ion_fin], outfile, ionfile)
        print(f"Filtered ionosphere written to: {outfile}")
    ## *******************  done  **********************

    # save final updated mask
    if wbd_use is not None:
        write_isce2_file(datasets=[~msk_drop.astype(np.int8)], outfile=maskout, ref_file=wbdfile_mlk)
        print(f"Ionophere filtering mask written to: {maskout}")

if __name__ == '__main__':
    '''
    Main driver.
    '''
    # Main Driver
    main()

