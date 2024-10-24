#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse.linalg import eigsh
from scipy.optimize import curve_fit
import astropy.constants as const
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import convolve2d
from scipy.stats import norm

# from scintools.scintools.ththmod import fft_axis, ext_find
from scintools.ththmod import fft_axis, ext_find

from Funcs_DP import *
from Funcs_DP_Orbsim import *

def regions_resampler_same_regions(idx, lensPos, delt, dyn, freq, sections):
    """
    Function to resample and extract the 2D FFT + congujate variables in the
    form of a list. Takes as inputs
    
    - ind: indeces indicating the regions of interest
    - lensPos: postion on the screen
    - delt: the step size to which you want to uniformly resample
    - dyn: dynamic spectrum matrix 
    - freq: frequency array from dynamic spectrum
    """
    
    res_dyn = []
    res_pos = []
    res_t = []
    res_fd =[]
    res_CS = []
    
    res_ts =[]
    res_dyns =[]
    res_fds =[]
    res_CSs = []


    for i in range(len(idx)):

        #do the resampling for each region
        #note position must be unitless
        pos_tmp, dyn_tmp = resampler( (lensPos[idx[i][0]: idx[i][1]].to(u.Mm)).value, 
                                          dyn[:, idx[i][0]: idx[i][1]], 
                                          delt )

        #stores position
        res_pos += [pos_tmp * u.Mm]
        #stores the resampled dynamic spectrum
        res_dyn += [dyn_tmp]
        #stores the position adjusted to time
        res_t += [( pos_tmp * u.Mm / (const.c / 1e4) ).to(u.s)]
        
        #computing 2D FFT and conjugate variables 
        fd_tmp, tau_tmp, cs_tmp = secondary_spectrum_noSS( ( pos_tmp * u.Mm / (const.c / 1e4) ).to(u.s) , 
                                                          freq, 
                                                          dyn_tmp, 
                                                          "time", 
                                                          "backward")
        #storing conjugate variables + FFT
        res_fd += [fd_tmp]
        res_CS += [cs_tmp]
        
    #computing all the values only for the regions that pass through the same part on the screen
    for i in range(len(sections)):
        
        index1, index2 = find_similar_regions(res_t[sections[i][0]].value , res_t[sections[i][1]].value )
        
        res_ts +=   [ [    res_t[sections[i][0]][index1], res_t[sections[i][1]][index2]     ] ]
        res_dyns += [ [res_dyn[sections[i][0]][:,index1], res_dyn[sections[i][1]][:,index2] ] ]
        
        #computing 2D FFT and conjugate variables 
        fd_tmp1, tau_tmp1, cs_tmp1 = secondary_spectrum_noSS( res_t[sections[i][0]][index1] , 
                                                          freq, 
                                                          res_dyn[sections[i][0]][:,index1] , 
                                                          "time", 
                                                          "backward")
        
        fd_tmp2, tau_tmp2, cs_tmp2 = secondary_spectrum_noSS( res_t[sections[i][1]][index2] , 
                                                          freq, 
                                                          res_dyn[sections[i][1]][:,index2] , 
                                                          "time", 
                                                          "backward")
        #storing conjugate variables + FFT
        res_fds += [[fd_tmp1, fd_tmp2]]
        res_CSs += [[cs_tmp1, cs_tmp2]]
    
    #return resampled time, position, dynamic spectrum, geometric delay, resampled doppler delay, resampled conjugate spec
    return res_t, res_pos, res_dyn, tau_tmp, res_fd, res_CS, res_ts, res_dyns, res_fds, res_CSs 

def minimal_stepsize_calculator(t0, nu0, freq0, 
                                 phase0, A_mutipliers_array0, phase_storage0):
    """
    Function to calculate an array with the minimal stepsizes needed for the fastest scintile of resampling
    """
    grand_delt0 = []

    for ok in range(len(A_mutipliers_array0)):


        for dj in range(len(phase_storage0[ok])):

            #getting the position information directly from given parameters
            #----------------------------------------------------------------------------------------

            lensPos0 = Ad_projection_unitless(t = t0.to(u.hour).value,
                                           nu = nu0.value.astype(np.float64),
                                           phase = phase0.value.astype(np.float64), 
                                           A = A_mutipliers_array0[ok] , 
                                           delta = phase_storage0[ok][dj] ) * 100 * u.Mm


            #set the position array
            y0 = (lensPos0.value) * u.Mm

            #set the separation for resampling array
            delt_tmp = np.min( np.abs( np.diff( y0.value ) ) ) 
            
            #storing the smallest stepsize to resample
            grand_delt0 += [delt_tmp]

        
    return grand_delt0
    
    

def iterator_similar_regions( sections_array0, dyn20, t0, nu0, freq0, 
                             phase0, A_mutipliers_array0, phase_storage0, 
                             delt0):
    
    """
    Function to compute the resampled 
    time (position in Mm rescaled by 1e4/c) grand_t0, 
    dynspec grand_dyn0, 
    doppler shift grand_fd0, 
    conjugate spectrum grand_CS0, 
    minimal stepsize for all parametes grand_delt0, 
    differential delay res_tau0
    
    for the regions that overlap in position given sections_array0
    for all the params A and delta given by A_multipliers_array0 and phase_storage0
    """
    
    #defining variable arrays to store all the resmapled params
    grand_t0 =[]
    grand_dyn0 =[]
    grand_fd0 = []
    grand_CS0 = []
    grand_delt0 = []


    for ok in range(len(A_mutipliers_array0)):

        
        local_t0 =[]
        local_dyn0 = []
        local_fd0 = []
        local_CS0 = []

        for dj in range(len(phase_storage0[ok])):

            #getting the position information directly from given parameters
            #----------------------------------------------------------------------------------------

            lensPos0 = Ad_projection_unitless(t = t0.to(u.hour).value,
                                           nu = nu0.value.astype(np.float64),
                                           phase = phase0.value.astype(np.float64), 
                                           A = A_mutipliers_array0[ok] , 
                                           delta = phase_storage0[ok][dj] ) * 100 * u.Mm

            #getting the indeces for the peaks+troughs+first/last points for the position on the screen
            idx, idx2 = peaks(lensPos0)


            #get the indeces of the different split regions
            #without last point included
            idx3 = generate_n_minus_1_x_2_array(idx2)        


            #doing the resampling
            #----------------------------------------------------------------------------------------

            #set the position array
            y0 = (lensPos0.value) * u.Mm

            #set the separation for resampling array
            delt_tmp = np.min( np.abs( np.diff( y0.value ) ) ) 
            
            #storing the smallest stepsize to resample
            grand_delt0 += [delt_tmp]

            #resample and compute conjugate variables
            res_t0, res_pos0, res_dyn0, res_tau0, res_fd0, res_CS0, res_ts0, res_dyns0, res_fds0, res_CSs0  = regions_resampler_same_regions(
                                                                                                  idx = idx3, 
                                                                                                  lensPos = y0, 
                                                                                                  delt = delt0, 
                                                                                                  dyn = dyn20, 
                                                                                                  freq = freq0,
                                                                                                  sections = sections_array0
                                                                                                    )
            local_t0 += [res_ts0]
            local_dyn0 += [res_dyns0]
            local_fd0 += [res_fds0]
            local_CS0 += [res_CSs0]

                
        grand_t0 += [local_t0]
        grand_dyn0 += [local_dyn0]
        grand_fd0 += [local_fd0]
        grand_CS0 += [local_CS0]
        
    return [grand_t0, grand_dyn0, grand_fd0, grand_CS0, grand_delt0, res_tau0]


def iterator_similar_regions_Ado( sections_array0, dyn20, t0, nu0, freq0, 
                             phase0, A_mutipliers_array0, phase_storage0, 
                             delt0):
    
    """
    Function to compute the resampled 
    time (position in Mm rescaled by 1e4/c) grand_t0, 
    dynspec grand_dyn0, 
    doppler shift grand_fd0, 
    conjugate spectrum grand_CS0, 
    minimal stepsize for all parametes grand_delt0, 
    differential delay res_tau0
    
    for the regions that overlap in position given sections_array0
    for all the params A and delta given by A_multipliers_array0 and phase_storage0
    """
    
    #defining variable arrays to store all the resmapled params
    grand_t0 =[]
    grand_dyn0 =[]
    grand_fd0 = []
    grand_CS0 = []
    grand_delt0 = []


    for ok in range(len(A_mutipliers_array0)):

        
        local_t0 =[]
        local_dyn0 = []
        local_fd0 = []
        local_CS0 = []

        for dj in range(len(phase_storage0[ok])):

            #getting the position information directly from given parameters
            #----------------------------------------------------------------------------------------

            lensPos0 = Ado_projection_unitless(t = t0.to(u.hour).value,
                                           nu = nu0.value.astype(np.float64),
                                           phase = phase0.value.astype(np.float64), 
                                           A = A_mutipliers_array0[ok] , 
                                           do = phase_storage0[ok][dj] ) * 100 * u.Mm

            #getting the indeces for the peaks+troughs+first/last points for the position on the screen
            idx, idx2 = peaks(lensPos0)


            #get the indeces of the different split regions
            #without last point included
            idx3 = generate_n_minus_1_x_2_array(idx2)        


            #doing the resampling
            #----------------------------------------------------------------------------------------

            #set the position array
            y0 = (lensPos0.value) * u.Mm

            #set the separation for resampling array
            delt_tmp = np.min( np.abs( np.diff( y0.value ) ) ) 
            
            #storing the smallest stepsize to resample
            grand_delt0 += [delt_tmp]

            #resample and compute conjugate variables
            res_t0, res_pos0, res_dyn0, res_tau0, res_fd0, res_CS0, res_ts0, res_dyns0, res_fds0, res_CSs0  = regions_resampler_same_regions(
                                                                                                  idx = idx3, 
                                                                                                  lensPos = y0, 
                                                                                                  delt = delt0, 
                                                                                                  dyn = dyn20, 
                                                                                                  freq = freq0,
                                                                                                  sections = sections_array0
                                                                                                    )
            local_t0 += [res_ts0]
            local_dyn0 += [res_dyns0]
            local_fd0 += [res_fds0]
            local_CS0 += [res_CSs0]

                
        grand_t0 += [local_t0]
        grand_dyn0 += [local_dyn0]
        grand_fd0 += [local_fd0]
        grand_CS0 += [local_CS0]
        
    return [grand_t0, grand_dyn0, grand_fd0, grand_CS0, grand_delt0, res_tau0]



def similar_region_equalizer(data_t, data_dyn, data_freq, sections):
    """
    Function that takes as inputs all the computed times, and dynamic spectrums for all values of A and delta that
    one is sweeping over, it looks for the minimum number of pixels that a given iteration of (A,delta) would have for all regions
    then makes all of the other data have the same number of pixels
    """
    
    #Getting the arrays of minimum points neede per region
    N_min = np.zeros(len(sections))
    
    for sec in range(len(N_min)):
    
        N_tmp = np.zeros(len(data_t))
        for i in range(len(N_tmp)):

            N_tmp2 = np.zeros(len(data_t[i]))
            for k in range(len(N_tmp2)):
                N_tmp2[k] = data_t[i][k][sec][0].shape[0]
            
            N_tmp[i] = np.min(N_tmp2)
        
        N_min[sec] = np.min(N_tmp)
    
    #Now to return a new array of portions with similar regions + same number of points
    data_t2 = []
    data_dyn2 = []
    data_fd2 = []
    data_cs2 = []
    
    for i in range(len(data_t)):
        
        data_t2_local = []
        data_dyn2_local = []
        data_fd2_local = []
        data_cs2_local = []
        
        for j in range(len(data_t[i])):
            
            data_t3_local = []
            data_dyn3_local = []
            data_fd3_local = []
            data_cs3_local = []
            
            for k in range(len(data_t[i][j])):
                
                a1 = 1.
                a2 = 1.
                
#                 if sections[k][0] % 2 != 0:
#                     a1 = -1.
#                 elif sections[k][1] % 2 != 0:
#                     a2 = -1.

                data_t3_local += [ [    a1 * data_t[i][j][k][0][- int(N_min[k]) :] , a2 * data_t[i][j][k][1][: int(N_min[k])]    ] ]
                data_dyn3_local += [[   data_dyn[i][j][k][0][:,- int(N_min[k]) :], data_dyn[i][j][k][1][:, : int(N_min[k])]   ]]
                
                fd_tmp1, tau_tmp1, cs_tmp1 = secondary_spectrum_noSS( a1 * data_t[i][j][k][0][- int(N_min[k]) :] , 
                                                          data_freq, 
                                                          data_dyn[i][j][k][0][:,- int(N_min[k]) :] , 
                                                          "time", 
                                                          "backward")
                
                fd_tmp2, tau_tmp2, cs_tmp2 = secondary_spectrum_noSS( a2 * data_t[i][j][k][1][: int(N_min[k])] , 
                                                          data_freq, 
                                                          data_dyn[i][j][k][1][:, : int(N_min[k])]  , 
                                                          "time", 
                                                          "backward")
                data_fd3_local += [[ fd_tmp1, fd_tmp2 ]]
                data_cs3_local += [[ cs_tmp1, cs_tmp2 ]]
                
                
            data_t2_local += [data_t3_local]
            data_dyn2_local += [data_dyn3_local]
            data_fd2_local += [data_fd3_local]
            data_cs2_local += [data_cs3_local]
            
        data_t2 += [data_t2_local]
        data_dyn2 += [ data_dyn2_local]
        data_fd2 += [data_fd2_local]
        data_cs2 += [data_cs2_local]
        
    return [data_t2, data_dyn2, data_fd2, data_cs2, tau_tmp1]

def vertical_rolling_average(A, b):
    # Define the vertical kernel for convolution
    kernel = np.ones((b, 1), dtype=np.float64) / b
    
    # Apply convolution along the rows (vertical direction)
    result = convolve2d(A, kernel, mode='same', boundary='symm')
    
    return result

def chi_similar_region(data_fd, data_cs, sstrength, data_tau, tau_min, tau_max, fdlims, fdeval, N_int, alpha):
    """
    Function to take two pairs of regions from data in 
    fd data_fd
    taud data_tau
    and conjugate spectrum data_cs
    
    compute the secondary spectrum for each region, scale each region by max(|CS|)^alpha
    apply a rolling average filter with sstrengh (sstrength indicates how many rows to average)
    crop the secondary spectrum by tau_min and tau_max
    interpolate and crop the secondary spectra in fd in (-fdeval, fdeval) with N_int elements
    
    return sum( (Sspec_i - Sspec_j)^2) and the correspodning fd, taud, and Sspec for both regions ij
    """
    
    
    chi_tmp = []
    plotter_tmp = []
    
    #loop over A
    for i in range(len(data_fd)):
        
        #loop over the phase
        chi_tmp_phase = []
        plotter_tmp_phase = []

        for j in range(len(data_fd[i])):
        
            regions_tmp = np.zeros(len(data_fd[0][0]))
            
            plotter_tmp2 = []
            
            #loop over the different regions
            for k in range(len(data_fd[0][0])):

                fd1_tmp = data_fd[i][j][k][0]
                fd2_tmp = data_fd[i][j][k][1]

                ss1_tmp = (np.abs(data_cs[i][j][k][0])**2) / np.max(np.abs(data_cs[i][j][k][0])**alpha)
                ss2_tmp = (np.abs(data_cs[i][j][k][1])**2) / np.max(np.abs(data_cs[i][j][k][1])**alpha)

                ss1 = vertical_rolling_average(ss1_tmp, sstrength)
                ss2 = vertical_rolling_average(ss2_tmp, sstrength)

                index = find_indices_within_range(data_tau.value, tau_min, tau_max )


                SS_int1, fd_int = interpolate_and_evaluate(x0 = fd1_tmp, 
                                                         SS = ss1[index] , #* data_tau[index, np.newaxis].value, 
                                                         xlims = fdlims, 
                                                         xint_range = fdeval, 
                                                         N = N_int,
                                                         kind = 'quadratic')

                SS_int2, fd_int = interpolate_and_evaluate(x0 = fd2_tmp, 
                                                         SS = ss2[index] ,# * data_tau[index, np.newaxis].value, 
                                                         xlims = fdlims,
                                                         xint_range = fdeval,
                                                         N = N_int,
                                                         kind = 'quadratic')

                #-----------------------------------------------------------------------------------------
    #             SS_int1 -= np.median(SS_int1)
#                 SS_int1 /= np.sum(SS_int1)

    #             SS_int2 -= np.median(SS_int2)
#                 SS_int2 /= np.sum(SS_int2)    
                
                #regions_tmp[k] = np.sum( np.abs( SS_int1 - np.flip(SS_int2,1) )**2)
                regions_tmp[k] = np.sum( np.abs( SS_int1 - SS_int2 )**2)
                plotter_tmp2 += [[fd_int,   data_tau[index], SS_int1, SS_int2]]
                
            chi_tmp_phase += [regions_tmp]
            plotter_tmp_phase += [plotter_tmp2 ]
        
        chi_tmp += [ chi_tmp_phase ]
        plotter_tmp += [plotter_tmp_phase]
        
    return chi_tmp, plotter_tmp