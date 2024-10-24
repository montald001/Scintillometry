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

# from scintools.scintools.ththmod import fft_axis, ext_find
# from scintools.scintools.scint_utils import is_valid, centres_to_edges
# import scintools.scintools.ththmod as thth

from scintools.ththmod import fft_axis, ext_find
from scintools.scint_utils import is_valid, centres_to_edges
import scintools.ththmod as thth

from Funcs_DP import *
from Funcs_DP_Orbsim import *


def secspec_linearizer( t, dyns, freqs, eta, delmax, plot = False, cutmid=5, startbin=3  ):
    
    dspec = 0
    timet = 0
    
    if (t[1] -  t[0] < 0):
        dspec= np.flip(dyns, 1) 
        timet= np.flip( t.astype(np.float64) )
    else:
        dspec= dyns
        timet = t.astype(np.float64)
    
    #loading a dynamicspec object
    bDyne = BasicDyn(
    name="AR",
    header=["AR"],
    times=timet.value,
    freqs=freqs.value,
    dyn=dspec,
    nsub=timet.shape[0],
    nchan=freqs.shape[0],
    dt=(timet[1] - timet[0]).value,
    df=(freqs[1] - freqs[0]).value,
    )
    
    #putting our data object into scintools format 
    dyn = Dynspec()
    dyn.load_dyn_obj(dyn=bDyne, verbose = False, process=False)
    
    dyn.norm_sspec(lamsteps=False, delmax = delmax, eta = eta, plot=plot, cutmid=cutmid, startbin=startbin )
    
    zzz = np.copy(dyn.normsspec)
    np.ma.set_fill_value(zzz, np.nan)
    
    fdopedges = centres_to_edges(dyn.normsspec_fdop)
    tdeledges = centres_to_edges(dyn.normsspec_tdel)
    
    medval = np.median(dyn.sspec[is_valid(dyn.sspec)*np.array(np.abs(dyn.sspec) > 0)])
    maxval = np.max(dyn.sspec[is_valid(dyn.sspec)*np.array(np.abs(dyn.sspec) > 0)])
    vmin = medval - 3
    vmax = maxval - 3
    
    return [dyn.normsspec_fdop, dyn.normsspecavg ], [fdopedges, tdeledges, zzz, vmin, vmax]
        
        
def resampler_given_Ad( A, c, time0, nu, phase, dyn2, freq, delt ):
        "Function to resample and plot sections given the parameters A, c, time and phase"
    
        lensPos = Ad_projection_unitless(t = time0.to(u.hour).value,
                                         nu = nu,
                                       phase = phase.value.astype(np.float64), 
                                       A = A , 
                                       delta = c )  * 100 *u.Mm
        
        #getting the indeces for the peaks+troughs+first/last points for the position on the screen
        idx, idx2 = peaks(lensPos)


        #get the indeces of the different split regions
        #without last point included
        idx3 = generate_n_minus_1_x_2_array(idx2)        
        
        #doing the resampling
        #----------------------------------------------------------------------------------------
        
        #set the position array
        y0 = (lensPos.value) * u.Mm
        
#         #set the separation for resampling array
#         delt =  np.min( np.abs( np.diff( y0.value ) ) ) * 0.5

        #resample and compute conjugate variables
        res_t, res_pos, res_dyn, res_tau, res_fd, res_CS  = regions_resampler(idx = idx3, 
                                                                              lensPos = y0, 
                                                                              delt = delt, 
                                                                              dyn = dyn2, 
                                                                              freq = freq)
        
         
        
        #plot the computed dynamic spectra + secondary spectra
        res_tau = res_tau.to(u.ms)
        
        return res_t, res_pos, res_dyn, res_tau, res_fd, res_CS
    
def resampler_given_Ad2( A, c, time0, nu, phase, dyn2, freq, Nres ):
        "Function to resample and plot sections given the parameters A, c, time and phase"
    
        lensPos = Ad_projection_unitless(t = time0.to(u.hour).value,
                                         nu = nu,
                                       phase = phase.value.astype(np.float64), 
                                       A = A , 
                                       delta = c )  * 100 *u.Mm
        
        #getting the indeces for the peaks+troughs+first/last points for the position on the screen
        idx, idx2 = peaks(lensPos)


        #get the indeces of the different split regions
        #without last point included
        idx3 = generate_n_minus_1_x_2_array(idx2)        
        
        #doing the resampling
        #----------------------------------------------------------------------------------------
        
        #set the position array
        y0 = (lensPos.value) * u.Mm
        
#         #set the separation for resampling array
#         delt =  np.min( np.abs( np.diff( y0.value ) ) ) * 0.5

        #resample and compute conjugate variables
        res_t, res_pos, res_dyn  = regions_resampler2(idx = idx3, 
                                                      lensPos = y0, 
                                                      Nres = Nres, 
                                                      dyn = dyn2, 
                                                      freq = freq)
        
         
        
        return res_t, res_pos, res_dyn
    
    
    
def resampler_given_Ado( A, do, time0, nu, phase, dyn2, freq, delt ):
        "Function to resample and plot sections given the parameters A, c, time and phase"
    
        lensPos = Ado_projection_unitless(t = time0.to(u.hour).value,
                                         nu = nu,
                                       phase = phase.value.astype(np.float64), 
                                       A = A , 
                                       do = do )  * 100 *u.Mm
        
        #getting the indeces for the peaks+troughs+first/last points for the position on the screen
        idx, idx2 = peaks(lensPos)


        #get the indeces of the different split regions
        #without last point included
        idx3 = generate_n_minus_1_x_2_array(idx2)        
        
        #doing the resampling
        #----------------------------------------------------------------------------------------
        
        #set the position array
        y0 = (lensPos.value) * u.Mm
        
#         #set the separation for resampling array
#         delt =  np.min( np.abs( np.diff( y0.value ) ) ) * 0.5

        #resample and compute conjugate variables
        res_t, res_pos, res_dyn, res_tau, res_fd, res_CS  = regions_resampler(idx = idx3, 
                                                                              lensPos = y0, 
                                                                              delt = delt, 
                                                                              dyn = dyn2, 
                                                                              freq = freq)
        
         
        
        #plot the computed dynamic spectra + secondary spectra
        res_tau = res_tau.to(u.ms)
        
        return res_t, res_pos, res_dyn, res_tau, res_fd, res_CS
    
def ss_2_thth(taud, fd, etad ):
    
    """
    Function to map conjugate spectrum coords in tau and fd to theta theta space given a curvature
    Note: 
    dim[taud] = [T]
    dim[fd] = [T]^-1
    dim[etad] = [T]^3
    
    """
    
    th1 = 0.5 * ( taud / (etad * fd) + fd)
    th2 = 0.5 * ( taud / (etad * fd) - fd)
    
    return np.array( (th1.to(u.mHz).value, th2.to(u.mHz).value) ) * u.mHz

def onclick(event):
    
    """
    Function to register the click of an image in matplotlib and
    then return a set of tuples with the coordinates 
    given the extent of the previous fig
    """
    # Check if the click is within the axes
    if event.inaxes:
        # Get the x and y coordinates in the image's extent
        ix, iy = event.xdata, event.ydata
        # Store the selected points
        selected_points.append((ix, iy))
        print(f'Selected point: x={ix}, y={iy}')
        # Optionally: Plot the selected point
        ax.plot(ix, iy, 'o')
        fig.canvas.draw()

        
def GeneralRebin(arr,xCoord,yCoord,rebinX=1,rebinY=1):
    arr2 = np.reshape(
    arr[: rebinY * (arr.shape[0] // rebinY), : rebinX * (arr.shape[1] // rebinX)],
    (arr.shape[0] // rebinY, rebinY, arr.shape[1] // rebinX, rebinX),
).mean((1, 3))
    xCoord2 = np.reshape(xCoord[:rebinX*arr2.shape[1]],(arr2.shape[1],rebinX)).mean(1)
    yCoord2 = np.reshape(yCoord[:rebinY*arr2.shape[0]],(arr2.shape[0],rebinY)).mean(1)
    return(arr2,xCoord2,yCoord2)


def tt_mapper2( dyn0, eta ):

    
    time2 = dyn0.times*u.s
    freq2= dyn0.freqs*u.MHz

    dspec2=np.copy(dyn0.dyn)
    mn = np.nanmean(dspec2)
    dspec2-=mn
    
    dspec_pad=np.pad(np.nan_to_num(dspec2),((0,dyn0.npad*dyn0.cwf),(0,dyn0.npad*dyn0.cwt)),mode='constant',constant_values=0)
    
    CS = np.fft.fftshift(np.fft.fft2(dspec_pad))
    tau = thth.fft_axis(freq2,u.us, dyn0.npad)
    fd = thth.fft_axis(time2,u.mHz, dyn0.npad)
    
    edges0 = dyn0.edges*(freq2.mean()/dyn0.fref)
    
    
    return CS, tau, fd, edges0


def data_2_thth(ts, fs, ds, eta, edg, ifmod, rebinN = 8):

    dspec = 0
    timet = 0

    if (ts[1] -  ts[0] < 0):
        dspec= np.flip( ds, 1) 
        timet= np.flip( ts.astype(np.float64) )
    else:
        dspec= ds
        timet = ts.astype(np.float64)

    dspec,timet,freq = GeneralRebin(dspec,timet,fs,rebinX=rebinN)
    dspec -= np.nanmean(dspec)
    
    #compute conjugate variables
    csu = np.fft.fftshift(np.fft.fft2( dspec ))
    tauu = thth.fft_axis(freq ,u.us, 0)
    fdu = thth.fft_axis(timet, u.mHz, 0)
    
    #compute thth map
    hh = thth.thth_map(CS = csu, 
         tau = tauu, 
         fd = fdu, 
         eta = eta, 
         edges = edg, 
         )
    
    #if you want thth modeler variables
    if ifmod == 1:
        #compute model thth and
        thth_red, thth2_red, recov, model, edges_red, w, V = thth.modeler(csu, 
                                                                          tauu, 
                                                                          fdu, 
                                                                          eta, 
                                                                          edg)
        return [ [fdu, tauu, csu, hh, edg, dspec, timet], [thth_red, thth2_red, recov, model, edges_red, w, V] ]
    else:
        return [fdu, tauu, csu, hh, edg, dspec, timet]

    
    
    
def data_2_svd(
    CS, tau, fd, eta, edges, etaArclet, edgesArclet, centerCut
):
    tau = thth.unit_checks(tau, "tau", u.us)
    fd = thth.unit_checks(fd, "fd", u.mHz)
    eta = thth.unit_checks(eta, "eta", u.s**3)
    edges = thth.unit_checks(edges, "edges", u.mHz)
    etaArclet = thth.unit_checks(etaArclet, "etaArclet", u.s**3)
    edgesArclet = thth.unit_checks(edgesArclet, "edgesArclet", u.mHz)
    centerCut = thth.unit_checks(centerCut, "Center Cut", u.mHz)

    thth_red, edges_red1, edges_red2 = thth.two_curve_map(
        CS, tau, fd, eta, edges, etaArclet, edgesArclet
    )
    cents1 = (edges_red1[1:] + edges_red1[:-1]) / 2
    thth_red[:, np.abs(cents1) < centerCut] = 0
    U, S, W = np.linalg.svd(thth_red)
    return U, S, W    

def data_2_2thth(
    CS, tau, fd, eta, edges, etaArclet, edgesArclet, centerCut
):
    tau = thth.unit_checks(tau, "tau", u.us)
    fd = thth.unit_checks(fd, "fd", u.mHz)
    eta = thth.unit_checks(eta, "eta", u.s**3)
    edges = thth.unit_checks(edges, "edges", u.mHz)
    etaArclet = thth.unit_checks(etaArclet, "etaArclet", u.s**3)
    edgesArclet = thth.unit_checks(edgesArclet, "edgesArclet", u.mHz)
    centerCut = thth.unit_checks(centerCut, "Center Cut", u.mHz)

    thth_red, edges_red1, edges_red2 = thth.two_curve_map(
        CS, tau, fd, eta, edges, etaArclet, edgesArclet
    )

    return thth_red, edges_red1, edges_red2


def asym_cacl(params):
    # Calculate Reduced TH-TH and largest eigenvalue/vector pair
    
    thth_red1, thth2_red1, recov1, model1, edges_red1, w1, V1 = params
    cents = (edges_red1[1:] + edges_red1[:-1]) / 2
    leftV = V1[: (cents.shape[0] - 1) // 2]
    rightV = V1[1 + (cents.shape[0] - 1) // 2:]
    asymm = (np.sum(np.abs(leftV) ** 2) - np.sum(np.abs(rightV) ** 2)) / (
        np.sum(np.abs(leftV) ** 2) + np.sum(np.abs(rightV) ** 2) )
        
    return asymm

def ad_block(A_multipliers, phimin, phimax, Nphi, atol, a_guess, ts, nu, phase, spacing = 0.08):
    """
    Function to find and constrain the A-delta parameter space
    """
    
    phase_storage = []
    A_storage = []

    #assign a limit for the range of A (the part below can be skipped if you know the range of A_multipliers)

    #assign a limit on delta (in this case it was named phi)

    #assign a tolerante for the root finder in Ad_overlap_finder
    peak_tolerance = atol

    #hold A fixed, and iterate to find the allowed ranges for delta
    #then iterate over all A
    for i in range(len(A_multipliers)):

        phi1 = Ad_overlap_finder(peaks = a_guess, 
                         peak_widths = peak_tolerance,
                         t = ts.to(u.hour).value,
                         nu = nu,
                         phase = phase.value.astype(np.float64), 
                         A = A_multipliers[i], 
                         crange = np.linspace(phimin,phimax,Nphi))

        if phi1 != (0,0):
            phase_storage += [ phi1]
            A_storage  += [A_multipliers[i]]

    #------------ Plot the inferred results against the true value-----------------------------------
    for ai, (lower, upper) in zip(A_storage, phase_storage):
        plt.vlines(ai , lower , upper , colors='r', linewidth=3)

    # plt.ylim(-0.7, 0)
    plt.xlim(0, 15)

    plt.grid()

    plt.title("Allowed parameters ", fontsize = 18 )
    plt.ylabel('$\\delta$ values (rad) ', fontsize = 16)
    plt.xlabel('A values $(\\Omega_p)$  ', fontsize = 16)

    #split the delta value space into intervals of separation of "spacing" for all A
    #this array retunrs all the possible delta values to sweep over each A
    param_arr = param_space_array(Astor = A_storage, dstor = phase_storage, spacing = spacing )

    Ad_counter = 0.
    for i in range(len(A_storage)):
        for j in range(len(param_arr[i])):
            if i == 0 and j ==0:
                blabel = 'points sampled over A'
            else:
                blabel = None
            plt.plot(A_storage[i], param_arr[i][j], 'ko', markersize = 5, label = blabel)
            Ad_counter += 1.

    print(Ad_counter)

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()

    # Assign the sections to be compared-----------------  
    # sections_array = [(0,1), (1,2), (2,3), (3,4), (4,5)]

    A_mutipliers_array = np.array(A_storage)
    
    return A_storage, param_arr


def ado_block(A_multipliers, phimin, phimax, Nphi, atol, a_guess, ts, nu, phase, spacing = 0.08):
    """
    Function to find and constrain the A-delta parameter space
    """
    
    phase_storage = []
    A_storage = []

    #assign a limit for the range of A (the part below can be skipped if you know the range of A_multipliers)

    #assign a limit on delta (in this case it was named phi)

    #assign a tolerante for the root finder in Ad_overlap_finder
    peak_tolerance = atol

    #hold A fixed, and iterate to find the allowed ranges for delta
    #then iterate over all A
    for i in range(len(A_multipliers)):

        phi1 = Ado_overlap_finder(peaks = a_guess, 
                         peak_widths = peak_tolerance,
                         t = ts.to(u.hour).value,
                         nu = nu,
                         phase = phase.value.astype(np.float64), 
                         A = A_multipliers[i], 
                         crange = np.linspace(phimin,phimax,Nphi))

        if phi1 != (0,0):
            phase_storage += [ phi1]
            A_storage  += [A_multipliers[i]]

    #------------ Plot the inferred results against the true value-----------------------------------
    for ai, (lower, upper) in zip(A_storage, phase_storage):
        plt.vlines(ai , lower , upper , colors='r', linewidth=3)

    # plt.ylim(-0.7, 0)
    plt.xlim(0, 15)

    plt.grid()

    plt.title("Allowed parameters ", fontsize = 18 )
    plt.ylabel('$\\Delta \\Omega$ values (' + str(phimin.unit) + ')', fontsize = 16)
    plt.xlabel('A values (unitless)  ', fontsize = 16)

    #split the delta value space into intervals of separation of "spacing" for all A
    #this array retunrs all the possible delta values to sweep over each A
    param_arr = param_space_array(Astor = A_storage, dstor = phase_storage, spacing = spacing )

    Ad_counter = 0.
    for i in range(len(A_storage)):
        for j in range(len(param_arr[i])):
            if i == 0 and j ==0:
                blabel = 'points sampled over A'
            else:
                blabel = None
            plt.plot(A_storage[i], param_arr[i][j], 'ko', markersize = 5, label = blabel)
            Ad_counter += 1.

    print(Ad_counter)

    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.legend()

    # Assign the sections to be compared-----------------  
    # sections_array = [(0,1), (1,2), (2,3), (3,4), (4,5)]

    A_mutipliers_array = np.array(A_storage)
    
    return A_storage, param_arr



def norm_ss_plotter(data, flip = 1.):
    
    plt.pcolormesh(flip * data[0], data[1], np.ma.filled(data[2]),
                               vmin=data[3], vmax=data[4], linewidth=0,
                               rasterized=True, shading='auto')
    
def gaussian(xs, amplitude, mean, stddev):
    return amplitude * np.exp(-(xs - mean)**2 / (2 * stddev**2))


def measurement_est( a_arr, da_arr):
    N = len( a_arr)
    w = 1 / da_arr**2
    w_avg = np.sum(a_arr * w) / np.sum(w)
    
    return w_avg, np.sqrt(np.sum(da_arr**2))/ N


def asym_cacl(params):
    # Calculate Reduced TH-TH and largest eigenvalue/vector pair
    
    thth_red1, thth2_red1, recov1, model1, edges_red1, w1, V1 = params
    cents = (edges_red1[1:] + edges_red1[:-1]) / 2
    leftV = V1[: (cents.shape[0] - 1) // 2]
    rightV = V1[1 + (cents.shape[0] - 1) // 2:]
    asymm = (np.sum(np.abs(leftV) ** 2) - np.sum(np.abs(rightV) ** 2)) / (
        np.sum(np.abs(leftV) ** 2) + np.sum(np.abs(rightV) ** 2) )
        
    return asymm


def svd2(dspec):
    bad=np.invert(np.isfinite(dspec))
    bad[dspec==0]=True
    s = np.nanstd(dspec,1)
    good_f = (s>0)*np.isfinite(s)
    s = np.nanstd(dspec,0)
    good_t = (s>0)*np.isfinite(s)
    dspec_red=np.copy(dspec[good_f][:,good_t])
    dspec_red[np.invert(np.isfinite(dspec_red))]=np.nanmean(dspec_red)
    dspec_red/= thth.svd_model(dspec_red).real
    temp=np.ones((dspec_red.shape[0],dspec.shape[1]))
    temp[:,good_t]=dspec_red
    dspec2 = np.zeros(dspec.shape)
    dspec2[good_f]=temp
    dspec2[bad]=np.nan
    return(dspec2)


def wrap180(a):
    return (a + 0.5*u.cy) % (1*u.cy) - 0.5*u.cy