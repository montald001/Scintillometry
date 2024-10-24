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
from scintools.ththmod import fft_axis, ext_find

from Funcs_DP import *
from Funcs_DP_Sspec import *


##Functions for simulating orbital motion----------------------

def regions_resampler(idx, lensPos, delt, dyn, freq):
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
    
    #return resampled time, position, dynamic spectrum, geometric delay, resampled doppler delay, resampled conjugate spec
    return res_t, res_pos, res_dyn, tau_tmp, res_fd, res_CS 

def resampler2( lensPos, dyn, Nres):
    '''
    Function to resample the dynamic spectrum given 
    dyn as dynamic spectrum numpy array
    lensPos as position array on the screen corresponding to time0
    ds is the resampled stepsize for the new dynamic spectrum
    
    returns resampled dynamic spectrum numpy matrix dyn_new
    and its new corresponding position array position_res
    '''
    #position array of the region
    position_reg = np.copy(lensPos)
    
    #create new position array (uniform timestep)
    position_res = 0
    
    
    #value to store the sign (if it's ascending or descending in its position)
    sign_val = position_reg[1] - position_reg[0]
    
    ds = np.abs( position_reg[0] - position_reg[-1] ) / Nres
    
    #create new uniform array if position is increasing
    if (sign_val > 0):
        position_res = np.arange(position_reg[0], position_reg[-1], ds  )
    #create new uniform array if position is decreasing
    else:
        position_res = -np.arange(position_reg[0], position_reg[-1], -ds  )
        position_reg *= -1 
        

    #create empty array for info storage
    info_array = np.empty(len(position_res), dtype=object)
    info_array[:] = [list() for _ in range(len(position_res))]
    

    #condition that saves the index for when the stepsize is bigger 
    #that data stepsize
    a = 0
    #condition that tells the loop if the previous step was one
    #bigger than the data stepsize
    b = 0
    #fraction of the previous range that contributes to the resampling
    f = 0

    #iterate over all points in a region's position 
    for i in range(len(position_reg)-1):   
        
        #finding the indices inside each position interval
        ind = find_indices_resampling(position_res, position_reg[i], position_reg[i+1], np.abs(sign_val) )

        #case if ds is bigger than the spacing between data points
        if len(ind) < 1:
            a += 1
            b = 1
            continue

        #case if ds is bigger than the spacing between data points and then finds one interval    
        elif (b == 1) and (len(ind) == 1):

            w = (position_res[ind[-1]] - position_reg[i]) / ds

            for j in range(1, a + 1):
                frac = (position_reg[i-j+1] - position_reg[i-j])/ds
                info_array[ind[-1]] += [(i - j, frac)]

            info_array[ind[-1]] += [(i, w)]

            if f > 0:

                wf = (position_reg[i-a] - position_res[ind[0]-1] ) / ds
                info_array[ind[-1]] += [(i-a-1, wf)]

            a = 0
            b = 0
        #case if ds is bigger than the spacing between data points and then finds one interval    
        elif (b==1) and (len(ind) > 1):

            for j in range(1, a+1):
                frac = (position_reg[i-j+1] - position_reg[i-j])/ds
                info_array[ind[0]] += [(i-j,frac)]

            if f > 0:

                wf = (position_reg[i-a] - position_res[ind[0]-1] ) / ds
                info_array[ind[0]] += [(i-a-1, wf)]                    

            w0 = (position_res[ind[0]] - position_reg[i] ) / ds

            info_array[ind[0]] += [(i, w0)]


            for j in range(1, len(ind)):
                info_array[ind[j]] += [(i,1.0)]

            a = 0
            b = 0

        #case if ds encompasses multiple data points
        elif (b != 1) and (len(ind) > 1):

            w0 = ( position_res[ind[0]] - position_reg[i]) / ds
            info_array[ind[0]] += [(i,w0) ]


            if f > 0:
                wf = (position_reg[i-a] - position_res[ind[0]-1] ) / ds

                info_array[ind[0]] += [(i-1, wf)]

            for j in range(1, len(ind)):
                info_array[ind[j]] += [(i,1.0)]

        #case if ds only encompasses one data point  
        elif (b != 1) and (len(ind) == 1):

            w0 = ( position_res[ind[-1]] - position_reg[i]) / ds
            info_array[ind[0]] += [(i,w0) ]



            if f > 0:
                wf = (position_reg[i-a] - position_res[ind[0]-1] ) / ds

                info_array[ind[0]] += [(i-1, wf)]


        if (len(ind) != 0) and (f == 0):
            f = 1
        
        #if it's the last index, resmaple the remaining dynamic spectrum columns into the last column
        if (i == len(position_reg)-2):
            #switiching the empty first element to have an empty last element
            info_array = np.roll( info_array , -1)
            #get the remaining fraction for the last element drizzle 
            frac = (position_reg[-1] - position_res[-1])/ds
            info_array[-1] += [(i+1, 1-frac), (i, frac)]
             
      
    if (sign_val < 0):
        position_res *= -1

    
    #creating a storage resampled dynamic spectrum
    dyn_new = np.zeros((dyn.shape[0], len(position_res)), dtype = dyn.dtype)
    
    #loop to create the new dynamic spectra
    for i in range(len(position_res)):
        
        #create a column temp variable to add every piece of the dyncamic spectrum
        #col_tmp = np.zeros((dyn.shape[0],1))
        col_tmp = np.zeros_like( dyn[:,0:1] )
        
        for j in range(len(info_array[i])):

            #getting the value of the dyn spec and multply it by the vspec 
            col_tmp += (dyn[:, info_array[i][j][0]] * info_array[i][j][1]).reshape(-1,1)

        dyn_new[:, i] =  col_tmp.reshape(dyn.shape[0])
    
    #return the resampled position + dynamic spectrum
    return position_res, dyn_new

def regions_resampler2(idx, lensPos, Nres, dyn, freq):
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

    for i in range(len(idx)):

        #do the resampling for each region
        #note position must be unitless
        pos_tmp, dyn_tmp = resampler2( (lensPos[idx[i][0]: idx[i][1]].to(u.Mm)).value, 
                                          dyn[:, idx[i][0]: idx[i][1]], 
                                          Nres )

        #stores position
        res_pos += [pos_tmp * u.Mm]
        #stores the resampled dynamic spectrum
        res_dyn += [dyn_tmp]
        #stores the position adjusted to time
        res_t += [( pos_tmp * u.Mm / (const.c / 1e4) ).to(u.s)]

    
    #return resampled time, position, dynamic spectrum, geometric delay, resampled doppler delay, resampled conjugate spec
    return res_t, res_pos, res_dyn

def resampling_stitcher( time_array, dyn_array):
    """
    Function to concatenate the different regions after separately inducing orbital motion
    """
    t_tmp = np.copy(time_array[0])
    dyn_tmp = np.copy( dyn_array[0])
    dt = np.abs( time_array[0][1] - time_array[0][0] )
    
    t_tmp = np.abs( t_tmp)
    
    for i in range(1,len(time_array)): 
        t_new = np.abs(time_array[i] - time_array[i][0]) + t_tmp[-1] + dt
        t_tmp = np.concatenate([t_tmp, t_new])
        dyn_tmp = np.concatenate([ dyn_tmp, dyn_array[i]],1)
    
    return t_tmp, dyn_tmp

def position_inverter(t, lenspos, index):
    """
    Function to hget the inverse map between time - position after resampling
    as in it will create the map that would have re-scaled the position from the resampling
    and return the time/position from the inverse map such that if the resampling is used
    on the inverted params, it will return the original scales for time and position
    """
    y_tmp = np.copy(lenspos[index[0][0] : index[0][1]])
    a = 1.
    
    for i in range(1,len(index)):
        if i % 2:
            a = -1.
        else:
            a = 1.

        if i != len(index)-1 :
            y_tmp = np.concatenate( [ y_tmp, a * (lenspos[index[i][0] : index[i][1]] - lenspos[index[i][0]]) + y_tmp[-1] + np.abs(lenspos[index[i][0]+1] - lenspos[index[i-1][1]])] )
        else:
            y_tmp = np.concatenate( [ y_tmp, a * (lenspos[index[i][0] : index[i][1]+1] - lenspos[index[i][0]]) + y_tmp[-1] + np.abs(lenspos[index[i][0]+1] - lenspos[index[i-1][1]])] )

    t_tmp = np.copy(t[index[0][0] : index[0][1]])
    a = 1.
    for i in range(1,len(index)):

        if i % 2:
            a = -1.
        else:
            a = 1.

        if i != len(index)-1 :
            t_tmp = np.concatenate( [ t_tmp, a * (t[index[i][0] : index[i][1]] - t[index[i][0]]) + t_tmp[-1] + np.abs(t[index[i][0]+1] - t[index[i-1][1]]) ] )
        else:
            t_tmp = np.concatenate( [ t_tmp, a * (t[index[i][0] : index[i][1]+1] - t[index[i][0]]) + t_tmp[-1] + np.abs(t[index[i][0]+1] - t[index[i-1][1]])] )

    return t_tmp, y_tmp


def create_orbital_dynamic_spectrum( A, c, Om, nu, t, f, dyn, dt):
    """
    Function to create a warped dynamic spectrum given the original projection
    t is 1D array astropy units of time
    f is a 1D array astropy units of frequency
    dyn is a 2D array
    
    returns new time and dynamic spectrum arrays to be resampled
    """
    
    phase = (t.to(u.hour) * Om).value * u.rad
    
    ys = Ad_projection_unitless( (t.to(u.hour)).value, nu, phase, A, c)

    #getting indeces of the regions (a region being before and after a turning point)
    idx3 = generate_n_minus_1_x_2_array(peaks(ys)[1]) 
    
    tss, yss = position_inverter(t = t, lenspos = ys, index = idx3)
    yss *= t[-1] / np.max( yss )
    
    #Creating a new uniformly spaced inverse map (in resampled time)
    y_uni = np.linspace(0, yss[-1], yss.shape[0])
    
    def quadratic_interpolation_scipy(x, y, xnew):
        # Create a quadratic interpolation function
        interpolator = interp1d(x, y, kind='quadratic', fill_value='extrapolate')

        # Evaluate the interpolated values at xnew
        ynew = interpolator(xnew)

        return ynew

    ts_res = quadratic_interpolation_scipy(yss, tss, y_uni)
    
    #
    poss = ( ts_res * u.hour * const.c  / 1e4).to(u.Mm)
    delts = np.abs( np.min( np.diff( poss.value ) ) ) * dt
    
    ind, ind2 = peaks(poss)
    ind3 = generate_n_minus_1_x_2_array(ind2) 
    
    
    res_t1, res_pos1, res_dyn1, res_tau1, res_fd1, res_CS1  = regions_resampler(idx = ind3, 
                                                                                  lensPos = poss, 
                                                                                  delt = delts, 
                                                                                  dyn = dyn, 
                                                                                  freq = f)
    
    tres2, dres2 = resampling_stitcher( res_t1, res_dyn1)
    tres2 = tres2.to(u.hour) / tres2.to(u.hour)[-1] * t[-1]
    
    return tres2, dres2, res_t1, res_pos1, res_dyn1, res_tau1, res_fd1, res_CS1

def induce_mirror_motion( t, nu, dyn, A, phi, c):
    """
    Function to mirror the dynamic spectrum on the regions where it turns back 
    on the simulated screen
    
    returns new mirrored dynamic spectrum and time
    """
    proj = Ad_projection_unitless(t.value, 
                                  nu,
                                  phi,
                                   A,
                                   c)
    
    regions = generate_n_minus_1_x_2_array( peaks(proj)[1] )
    
    t_norm = t / np.max(t)
    proj_norm = (proj - np.min(proj)).value
    proj_norm /= np.max(proj_norm)
    
    new_dyn = []
    
    for i in range(len(regions)):
        
        
        reg_index = find_indices_inside_range(np.array([proj_norm[regions[i][0]], proj_norm[regions[i][1]]  ]), t_norm)
        
        
        if ( proj_norm[regions[i][1]] - proj_norm[regions[i][0]] ) < 0 :
            
            new_dyn += [np.flip(dyn[:,reg_index],1)]
        else:
            new_dyn += [dyn[:,reg_index]]
            
    new_dyn = np.concatenate(new_dyn, axis=1)
            
    return new_dyn, np.linspace(t[0], t[-1], new_dyn.shape[1])




def plotter_given_Ad2( A, c, time0, nu, phase, dyn2, freq, vmin, vmax, aaa, atol, fmax, dynmin, dynmax, taumax, delt, region_display = 0):
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
        
        for i in range(len(res_CS)):
            res_CS[i] /= np.sqrt( np.max( np.abs(res_CS[i])**2 )  )
            
        run_name = "A = " + d2str(A,2)+ " $\\Omega $" + ", $\\delta = $" + d2str(c,2)
        sections_plotter2(res_dyn = res_dyn, 
                         res_pos = res_pos, 
                         freq = freq, 
                         res_fd = res_fd, 
                         res_tau = res_tau, 
                         res_CS = res_CS , 
                         fmax = fmax, 
                         title = run_name, 
                         save = 0, 
                         path = "testpic",
                         display = 1,
                         vmin = vmin,
                         vmax = vmax,
                         taumax = taumax)
        plt.show()
        
        if region_display == 1 :
            plt.figure(figsize=(10,4))

            plt.subplot(2,1,1)
            plt.imshow(dyn2,
                       origin='lower',
                       aspect = 'auto',
            #            interpolation = 'nearest',
                       vmin = dynmin,
                       vmax = dynmax,
                       extent=ext_find(time0.to(u.hour), freq),
                      cmap = 'gray_r' )


            plt.ylabel('Frequency (MHz)')
            plt.xlabel([])

            # aaa = np.array([0.583, 1.847 + 0.03, 3.125, 4.375 - 0.0])

            for i in range(len( aaa) ):
                plt.axvline(x=aaa[i]+atol, c='r')
                plt.axvline(x=aaa[i]-atol, c='r')

            plt.xticks(np.arange(0, np.max( time0.to(u.hour).value ) ,0.5 ))
            plt.subplots_adjust(wspace=0, hspace=0)

            plt.subplot(2,1,2)
            plt.plot(time0.to(u.hour), lensPos / 100)
            plt.xlim([time0.to(u.hour)[0].value, time0.to(u.hour)[-1].value])
            for i in range(len( aaa) ):
                plt.axvline(x=aaa[i]+atol, c='r')
                plt.axvline(x=aaa[i]-atol, c='r')

            plt.xlabel('Time (hours)')
            
            plt.show()
            
            # Calculate relative widths based on the size of x_data
            relative_widths = [np.abs(x[0] - x[-1]).value for x in res_pos]
            total_width = sum(relative_widths)
            relative_widths = [width / total_width for width in relative_widths]

            # Create a figure with subplots
            fig, axes = plt.subplots(1, len(res_pos), figsize=(sum(relative_widths) * 12, 8), 
                                     gridspec_kw={'width_ratios': relative_widths,'wspace': 0 })



            # Plot data for each subplot
            for i, (x, ax) in enumerate(zip(res_pos, axes)):
                ax.imshow(res_dyn[i],
                         origin = 'lower',
                         aspect = 'auto',
                         interpolation = None,
                         extent = ext_find(res_pos[i], freq), 
                          vmin=dynmin,
                         vmax = dynmax)  # Replace this with your actual plot data
                ax.set_title(f'Region {i + 1}')
                fig.subplots_adjust(wspace=-10, hspace=0.35)
                if i ==0:
                    ax.set_ylabel('Frequency (MHz)')
                else:
                    ax.set_yticks([])
                if i == len(res_pos)//2:
                    ax.set_xlabel('Position on the screen (Mm)')



            plt.tight_layout()
            plt.show()
            
def plotter_given_Ad3( A, c, time0, nu, phase, dyn2, freq, vmin, vmax, aaa, atol, fmax, dynmin, dynmax, taumax, delt, sections_array, region_display = 0):
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
        
        for i in range(len(res_CS)):
            res_CS[i] /= np.sqrt( np.max( np.abs(res_CS[i])**2 )  )
            
        run_name = "A = " + d2str(A,2)+ " $\\Omega $" + ", $\\delta = $" + d2str(c,2)
        sections_plotter2(res_dyn = res_dyn, 
                         res_pos = res_pos, 
                         freq = freq, 
                         res_fd = res_fd, 
                         res_tau = res_tau.to(u.us), 
                         res_CS = res_CS , 
                         fmax = fmax, 
                         title = run_name, 
                         save = 0, 
                         path = "testpic",
                         display = 1,
                         vmin = vmin,
                         vmax = vmax,
                         taumax = taumax)
        plt.show()
        
        #plot for the different regions that overlap 
        #-----------------------------------------------------------------------------------------------------------------------------
        datas = iterator_similar_regions( sections_array0 = sections_array, 
                         dyn20 = dyn2, 
                         t0 = time0,
                         nu0 = nu,
                         freq0 = freq, 
                         phase0 = phase, 
                         A_mutipliers_array0 = [ A ], #A_mutipliers_array, 
                         phase_storage0 = [ [c] ], # param_arr,  
                         delt0 = delt)
        


        plt.figure(figsize=(15,15))
        for k in range(2):
            plt.subplot(len(sections_array),2,2*k+1)
            ssr1 = np.abs(datas[3][0][0][k][0] )**2
            ssr1 /= np.max(ssr1)
            ssr2 = np.abs(datas[3][0][0][k][1] )**2
            ssr2 /= np.max(ssr2)

            secondary_spectrum_plotter_time2( datas[2][0][0][k][0], 
                                             datas[5].to(u.us), 
                                             ssr1, 
                                             vmin, 
                                             vmax , 
                                             15, 
                                             None, 
                                             None)
            plt.ylim(0,taumax)
            plt.xlim(-fmax,fmax)
            
            plt.subplot(len(sections_array),2,2*k+2)
            secondary_spectrum_plotter_time2( datas[2][0][0][k][1], 
                                             datas[5].to(u.us), 
                                             ssr2, 
                                             vmin, 
                                             vmax , 
                                             15, 
                                             None, 
                                             None)
            plt.ylim(0.,taumax)
            plt.xlim(-fmax,fmax)
        
        
        #----------------------------------------------------------------------------------------------------------------------------
        #plot for the new dynamic spectrum
        if region_display == 1 :
            plt.figure(figsize=(10,4))

            plt.subplot(2,1,1)
            plt.imshow(dyn2,
                       origin='lower',
                       aspect = 'auto',
            #            interpolation = 'nearest',
                       vmin = dynmin,
                       vmax = dynmax,
                       extent=ext_find(time0.to(u.hour), freq),
                      cmap = 'gray_r' )


            plt.ylabel('Frequency (MHz)')
            plt.xlabel([])

            # aaa = np.array([0.583, 1.847 + 0.03, 3.125, 4.375 - 0.0])

            for i in range(len( aaa) ):
                plt.axvline(x=aaa[i]+atol, c='r')
                plt.axvline(x=aaa[i]-atol, c='r')

            plt.xticks(np.arange(0, np.max( time0.to(u.hour).value ) ,0.5 ))
            plt.subplots_adjust(wspace=0, hspace=0)

            plt.subplot(2,1,2)
            plt.plot(time0.to(u.hour), lensPos / 100)
            plt.xlim([time0.to(u.hour)[0].value, time0.to(u.hour)[-1].value])
            for i in range(len( aaa) ):
                plt.axvline(x=aaa[i]+atol, c='r')
                plt.axvline(x=aaa[i]-atol, c='r')

            plt.xlabel('Time (hours)')
            
            plt.show()
            
            # Calculate relative widths based on the size of x_data
            relative_widths = [np.abs(x[0] - x[-1]).value for x in res_pos]
            total_width = sum(relative_widths)
            relative_widths = [width / total_width for width in relative_widths]

            # Create a figure with subplots
            fig, axes = plt.subplots(1, len(res_pos), figsize=(sum(relative_widths) * 12, 8), 
                                     gridspec_kw={'width_ratios': relative_widths,'wspace': 0 })



            # Plot data for each subplot
            for i, (x, ax) in enumerate(zip(res_pos, axes)):
                ax.imshow(res_dyn[i],
                         origin = 'lower',
                         aspect = 'auto',
                         interpolation = None,
                         extent = ext_find(res_pos[i], freq), 
                          vmin=dynmin,
                         vmax = dynmax)  # Replace this with your actual plot data
                ax.set_title(f'Region {i + 1}')
                fig.subplots_adjust(wspace=-10, hspace=0.35)
                if i ==0:
                    ax.set_ylabel('Frequency (MHz)')
                else:
                    ax.set_yticks([])
                if i == len(res_pos)//2:
                    ax.set_xlabel('Position on the screen (Mm)')



            plt.tight_layout()
            plt.show()
            
            
            
def plotter_given_Ado3( A, do, time0, nu, phase, dyn2, freq, vmin, vmax, aaa, atol, fmax, dynmin, dynmax, taumax, delt, sections_array, region_display = 0):
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
        
        for i in range(len(res_CS)):
            res_CS[i] /= np.sqrt( np.max( np.abs(res_CS[i])**2 )  )
            
        run_name = "A = " + d2str(A,2)+ " $\\Omega $" + ", $\\delta = $" + d2str(do,2)
        sections_plotter2(res_dyn = res_dyn, 
                         res_pos = res_pos, 
                         freq = freq, 
                         res_fd = res_fd, 
                         res_tau = res_tau.to(u.us), 
                         res_CS = res_CS , 
                         fmax = fmax, 
                         title = run_name, 
                         save = 0, 
                         path = "testpic",
                         display = 1,
                         vmin = vmin,
                         vmax = vmax,
                         taumax = taumax)
        plt.show()
        
        #plot for the different regions that overlap 
        #-----------------------------------------------------------------------------------------------------------------------------
        datas = iterator_similar_regions_Ado( sections_array0 = sections_array, 
                         dyn20 = dyn2, 
                         t0 = time0,
                         nu0 = nu,
                         freq0 = freq, 
                         phase0 = phase, 
                         A_mutipliers_array0 = [ A ], #A_mutipliers_array, 
                         phase_storage0 = [ [do] ], # param_arr,  
                         delt0 = delt)
        


        plt.figure(figsize=(15,15))
        for k in range(2):
            plt.subplot(len(sections_array),2,2*k+1)
            ssr1 = np.abs(datas[3][0][0][k][0] )**2
            ssr1 /= np.max(ssr1)
            ssr2 = np.abs(datas[3][0][0][k][1] )**2
            ssr2 /= np.max(ssr2)

            secondary_spectrum_plotter_time2( datas[2][0][0][k][0], 
                                             datas[5].to(u.us), 
                                             ssr1, 
                                             vmin, 
                                             vmax , 
                                             15, 
                                             None, 
                                             None)
            plt.ylim(0,taumax)
            plt.xlim(-fmax,fmax)
            
            plt.subplot(len(sections_array),2,2*k+2)
            secondary_spectrum_plotter_time2( datas[2][0][0][k][1], 
                                             datas[5].to(u.us), 
                                             ssr2, 
                                             vmin, 
                                             vmax , 
                                             15, 
                                             None, 
                                             None)
            plt.ylim(0.,taumax)
            plt.xlim(-fmax,fmax)
        
        
        #----------------------------------------------------------------------------------------------------------------------------
        #plot for the new dynamic spectrum
        if region_display == 1 :
            plt.figure(figsize=(10,4))

            plt.subplot(2,1,1)
            plt.imshow(dyn2,
                       origin='lower',
                       aspect = 'auto',
            #            interpolation = 'nearest',
                       vmin = dynmin,
                       vmax = dynmax,
                       extent=ext_find(time0.to(u.hour), freq),
                      cmap = 'gray_r' )


            plt.ylabel('Frequency (MHz)')
            plt.xlabel([])

            # aaa = np.array([0.583, 1.847 + 0.03, 3.125, 4.375 - 0.0])

            for i in range(len( aaa) ):
                plt.axvline(x=aaa[i]+atol, c='r')
                plt.axvline(x=aaa[i]-atol, c='r')

            plt.xticks(np.arange(0, np.max( time0.to(u.hour).value ) ,0.5 ))
            plt.subplots_adjust(wspace=0, hspace=0)

            plt.subplot(2,1,2)
            plt.plot(time0.to(u.hour), lensPos / 100)
            plt.xlim([time0.to(u.hour)[0].value, time0.to(u.hour)[-1].value])
            for i in range(len( aaa) ):
                plt.axvline(x=aaa[i]+atol, c='r')
                plt.axvline(x=aaa[i]-atol, c='r')

            plt.xlabel('Time (hours)')
            
            plt.show()
            
            # Calculate relative widths based on the size of x_data
            relative_widths = [np.abs(x[0] - x[-1]).value for x in res_pos]
            total_width = sum(relative_widths)
            relative_widths = [width / total_width for width in relative_widths]

            # Create a figure with subplots
            fig, axes = plt.subplots(1, len(res_pos), figsize=(sum(relative_widths) * 12, 8), 
                                     gridspec_kw={'width_ratios': relative_widths,'wspace': 0 })



            # Plot data for each subplot
            for i, (x, ax) in enumerate(zip(res_pos, axes)):
                ax.imshow(res_dyn[i],
                         origin = 'lower',
                         aspect = 'auto',
                         interpolation = None,
                         extent = ext_find(res_pos[i], freq), 
                          vmin=dynmin,
                         vmax = dynmax)  # Replace this with your actual plot data
                ax.set_title(f'Region {i + 1}')
                fig.subplots_adjust(wspace=-10, hspace=0.35)
                if i ==0:
                    ax.set_ylabel('Frequency (MHz)')
                else:
                    ax.set_yticks([])
                if i == len(res_pos)//2:
                    ax.set_xlabel('Position on the screen (Mm)')



            plt.tight_layout()
            plt.show()
            
def sections_plotter2(res_dyn, res_pos, freq, res_fd, res_tau, res_CS, fmax, title, save, path, display, vmin, vmax, taumax):
    #function to plot and save the resampled dynamic spectra + secondary spectra for a given
    #set of parameters
    
    plt.figure(figsize = (10,14))
    for i in range(len(res_dyn)):
        plt.tight_layout()
        plt.suptitle(title, fontsize = 14)

        plt.subplot(len(res_dyn),2,2*i+1)

        xlabeld, xlabels = None, None
        ylabeld = "Frequency (MHz)"
        ylabels = "$\\tau_D \: (\\mu s)$"

        if ( i == len(res_dyn) -1):
            xlabeld = "Position (Mm)"
            xlabels = "$f_D $ (mHz)"

        plt.title("Region (" + str(i + 1) + ")")
        dynamic_spectrum_plotter(res_dyn[i], res_pos[i], freq, xlabeld, ylabeld,)


        plt.subplot(len(res_dyn),2,2*(i+1) )


        secondary_spectrum_plotter_time2( res_fd[i], res_tau, np.abs(res_CS[i])**2, vmin, vmax , fmax, xlabels, ylabels)
        plt.ylim(0., taumax)
        
    if save == 1:
        plt.savefig(path + ".png")
        
    if display != 1: 
        plt.close()