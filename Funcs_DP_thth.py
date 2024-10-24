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

from Funcs_DP import *

# import scintools.scintools.ththmod as thth
# from scintools.scintools.dynspec import BasicDyn, Dynspec
# from scintools.scintools.ththmod import fft_axis, ext_find

import scintools.ththmod as thth
from scintools.dynspec import BasicDyn, Dynspec
from scintools.ththmod import fft_axis, ext_find


def result_interpreter( ththresults, sample_datas):
    
    """
    Function to take in the list that followed from the parallelized
    computation of curvatures, and reshapes the curvatures into the same 
    shape as param_array which is an array containing the phases for 
    all A
    """
    # Calculate the dimensions of the sample data
    A_num = len(sample_datas)
    k_num = len(sample_datas[0][0])
    N_num = len(sample_datas[0][0][0])
    
     # Calculate the total number of curvatures to group over
    reg_num = k_num * N_num
    big_N = int( len(ththresults) / reg_num )
    
    
    etas = np.array( ththresults )[:,0]
    etas_err = np.array( ththresults )[:,1]
    
    res_tmp = []
    res_err_tmp = []
    
    #group them into a list with the same number of elements as the total number
    #of allowed delta values
    for dindex in range(big_N):
        res_tmp += [etas[0 + dindex * reg_num : (dindex) * reg_num + reg_num]]
        res_err_tmp += [etas_err[0 + dindex * reg_num : (dindex) * reg_num + reg_num]]
     
    res_tmp2 = []
    res_err_tmp2 = []
    
    ccounter = 0
    
    #reshape the list into the same shape as param_arr
    for aindex in range(A_num):
        res_tmp3 = []
        res_err_tmp3 = []
        for dindex in range(len(sample_datas[aindex])):
            res_tmp3 += [res_tmp[ccounter]]
            res_err_tmp3 += [res_err_tmp[ccounter]]
            ccounter += 1
        
        res_tmp2 += [res_tmp3]
        res_err_tmp2 += [res_err_tmp3]  
        
    return res_tmp2, res_err_tmp2

def thth_curvature_fitter( t, dyns, freqs, N, etamin, etamax, elims, fw, npad, tau_lim = 1000. * u.us, plots = 0):
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
    
    #prep dyn for thetatheta
    dyn.prep_thetatheta(#cwf=101,
#                     cwf = freqs.shape[0],
#                     cwt = fd.shape[0], 
                    edges_lim=elims,
                    eta_min=etamin,
                    eta_max=etamax,
                    nedge = N ,
                    verbose=False,
                    fw=fw,
                    tau_lim = tau_lim,
                    fitting_proc = 'incoherent',
                    npad = npad)
    
    #calculate curvature and error
#     ts = time.time()
    if plots == 0:
        dyn.fit_thetatheta()
        
        #in case of getting infinite retry params
        if np.isnan( ((dyn.ththeta).to(u.s**3)).value ) == True:
            
            for saver_ind in range(3):
                    #prep dyn for thetatheta
                dyn.prep_thetatheta(
                                edges_lim=elims,
                                eta_min=etamin,
                                eta_max=etamax,
                                nedge = N, #int( N * (1 + (saver_ind+1)/5) ),
                                verbose=False,
                                fw=fw * (3 - saver_ind)/4,
                                tau_lim = tau_lim,
                                fitting_proc = 'incoherent',
                                npad = npad)
                if np.isnan( ((dyn.ththeta).to(u.s**3)).value ) == True and saver_ind == 2:
                    return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
                
                dyn.fit_thetatheta()
                
                if np.isnan( ((dyn.ththeta).to(u.s**3)).value ) == False:
                    return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
        
        #in case of getting negative curvatures retry params
        elif ((dyn.ththeta).to(u.s**3)).value  < 0:
            
            for saver_ind in range(3):
                    #prep dyn for thetatheta
                dyn.prep_thetatheta(
                                edges_lim=elims,
                                eta_min=etamin,
                                eta_max=etamax,
                                nedge = N, #int( N * (1 + (saver_ind+1)/5) ),
                                verbose=False,
                                fw=fw * (3 - saver_ind)/4,
                                tau_lim = tau_lim,
                                fitting_proc = 'incoherent',
                                npad = npad)
                if ((dyn.ththeta).to(u.s**3)).value  < 0 and saver_ind == 2:
                    return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
                
                dyn.fit_thetatheta()
                
                if ((dyn.ththeta).to(u.s**3)).value  > 0:
                    return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
            
            
        else:
        
            return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
        
    else:
        dyn.thetatheta_single(cf=0, ct=0, verbose=True)
        
        
        

def thth_curvature_fitter_coherent( t, dyns, freqs, N, etamin, etamax, elims, fw, npad, tau_lim = 1000. * u.us, plots = 0):
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
    
    #prep dyn for thetatheta
    dyn.prep_thetatheta(#cwf=101,
#                     cwf = freqs.shape[0],
#                     cwt = fd.shape[0], 
                    edges_lim=elims,
                    eta_min=etamin,
                    eta_max=etamax,
                    nedge = N ,
                    verbose=False,
                    fw=fw,
                    tau_lim = tau_lim,
                    fitting_proc = 'standard',
                    npad = npad)
    
    #calculate curvature and error
#     ts = time.time()
    if plots == 0:
        dyn.fit_thetatheta()
        
        #in case of getting infinite retry params
        if np.isnan( ((dyn.ththeta).to(u.s**3)).value ) == True:
            
            for saver_ind in range(3):
                    #prep dyn for thetatheta
                dyn.prep_thetatheta(
                                edges_lim=elims,
                                eta_min=etamin,
                                eta_max=etamax,
                                nedge = N, #int( N * (1 + (saver_ind+1)/5) ),
                                verbose=False,
                                fw=fw * (3 - saver_ind)/4,
                                tau_lim = tau_lim,
                                fitting_proc = 'standard',
                                npad = npad)
                if np.isnan( ((dyn.ththeta).to(u.s**3)).value ) == True and saver_ind == 2:
                    return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
                
                dyn.fit_thetatheta()
                
                if np.isnan( ((dyn.ththeta).to(u.s**3)).value ) == False:
                    return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
        
        #in case of getting negative curvatures retry params
        elif ((dyn.ththeta).to(u.s**3)).value  < 0:
            
            for saver_ind in range(3):
                    #prep dyn for thetatheta
                dyn.prep_thetatheta(
                                edges_lim=elims,
                                eta_min=etamin,
                                eta_max=etamax,
                                nedge = N, #int( N * (1 + (saver_ind+1)/5) ),
                                verbose=False,
                                fw=fw * (3 - saver_ind)/4,
                                tau_lim = tau_lim,
                                fitting_proc = 'standard',
                                npad = npad)
                if ((dyn.ththeta).to(u.s**3)).value  < 0 and saver_ind == 2:
                    return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
                
                dyn.fit_thetatheta()
                
                if ((dyn.ththeta).to(u.s**3)).value  > 0:
                    return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
            
            
        else:
        
            return ((dyn.ththeta).to(u.s**3)).value, ((dyn.ththetaerr).to(u.s**3)).value
        
    else:
        dyn.thetatheta_single(cf=0, ct=0, verbose=True)