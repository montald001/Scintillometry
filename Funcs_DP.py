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

import os
from datetime import datetime

# from scintools.scintools.dynspec import BasicDyn, Dynspec
# from scintools.scintools.ththmod import fft_axis, ext_find

from scintools.dynspec import BasicDyn, Dynspec
from scintools.ththmod import fft_axis, ext_find


def dynamic_spectrum_extractor( day ):
    '''
    Function to extract the dynamic spectrum with frequency and time arrays
    day is the observation day in MJD
    '''

    #Get all the header info
    f = open(f'/fs/lustre/scratch/montalvo/research/P0737_obs/{day}/{day}.tiss5t')
    line = f.readline()
    param = line.split() 
    
    
    fc = float(param[0]) #bandcentre
    bw = float(param[1]) #bandwidth
    nchan = int(param[2]) #number of frequency channels
    #param[3] is a number of bins but I'm not sure how to use this information
    tstep = float(param[4]) #time step
    fs = int(param[5])-1 #start frequency channel
    fe = int(param[6])-1 #end frequency channel
    phase0 =  float(f.readline().split()[0])*u.deg #Orbital phase at start of observation
    
    
    #-----------------------------------------------
    # Set up time and frequency chunks
    #-----------------------------------------------
    
    
    #Load dynamic spectrum
    #use this in case file isn't inside the folder of the obs. day
    dspec = np.load(f'/fs/lustre/scratch/montalvo/research/P0737_obs/Rickett_{day}dspec.npy')
    
#     dspec_file = "/fs/lustre/scratch/montalvo/research/P0737_obs/" + str(day) + '/Rickett_' + str(day) + 'dspec.npy'
#     dspec = np.load(dspec_file) # Dynamic spectrum

    #Change channel/bin values into real physical values
    time = np.linspace(0,dspec.shape[1]*tstep,num=dspec.shape[1])*u.s
    freq = np.linspace(fc-0.5*bw,fc+0.5*bw, num=nchan)*u.MHz

    
#     #cut out the bad RFI data
    dyn = dspec
    dyn = dspec[fs:fe,:] 
    freq = freq[fs:fe]
    
    return time, freq, dyn


#function to subtract the dynamic spectrum's average over time for each frequency bin, it also scales and normalizes
#so its values are above 0 but it's constrained to 1 
#function to subtract the dynamic 
def dynamic_spectrum_averager(dyn):
    '''
    Function to normalize and average out the dynamic spectrum.
    Takes a numpy matrix for the dynamic spectrum
    Input
    dyn numpy array
    Outputs
    dyn2 numpy normalized dynamic spectrum
    '''
    n = dyn.shape[0]
    dyn2 = np.copy(dyn)

    dyn2 -= np.min(dyn2)
    dyn2 /= np.max(dyn2)
    
    for i in range(n):
        dyn2[i,:] /= np.mean( dyn2[i,:] )
    return dyn2


#function to return the central frequency range given an observation name mjObs
def central_freq(mjObs):
    if mjObs == 53560:
        return 1900 * u.MHz 
    else:
        return 820 * u.MHz


#function to get position-on-the-screen peaks and troughs as indeces
def peaks(lensPos):
    '''
    Finds the peaks and troughs at which the pulsar changes direction on the screen
    then returns the indeces that correspond to the turning point 
    lensPos is the Pulsar position on the screen
    idx are the indeces or turning points
    idx2 are the ideces from idx appended to the endpoints of the array
    
    this has the purpose of dividing the position on the screen into segments where
    the pulsar position doesn't overlap when going back on the screen
    '''
    #indeces of the peaks
    idx = np.argwhere(np.abs(np.diff(np.sign(np.diff(lensPos)))) > .5)[:,0] + 2
    
    #indeces of the peaks + first and last indeces
    idx2 = np.concatenate((np.zeros(1), idx, np.array([len(lensPos)-1]))).astype(int)
    
    return idx, idx2

def generate_n_minus_1_x_2_array(idx):
    """
    Generate a (n-1)x2 NumPy array based on the specified pattern.

    Parameters:
        idx (numpy.ndarray): Input NumPy array of length n.

    Returns:
        numpy.ndarray: A (n-1)x2 NumPy array following the specified pattern.
    """
    if len(idx) < 2:
        raise ValueError("Input array 'idx' must have at least 2 elements.")

    # Initialize the result array with zeros
    result = np.zeros((len(idx) - 1, 2), dtype=np.int32)

    for i in range(len(idx) - 1):
        result[i, 0] = idx[i] 
        result[i, 1] = idx[i+1]

    return result



#function to get the indices of an array given a range yrange
def find_indices_inside_range(yrange, y):
    """
    Find indices of elements in the input NumPy array y that are inside the specified range yrange.
    
    Parameters:
    yrange (tuple or list): A tuple or list representing the range (min_value, max_value).
    y (numpy.ndarray): Input NumPy array.
    
    Returns:
    numpy.ndarray: Indices of elements inside the specified range.
    """
    min_value = min(yrange) 
    max_value = max(yrange)
    # Find indices where y is inside the range
    indices = np.where((y >= min_value) & (y <= max_value))[0]
    return indices



#function to get the indices of an array given a range yrange
def find_indices_within_range(s, smin, smax):
    """
    Find the indices of elements in the array s that are within the range [smin, smax].

    Parameters:
        s (numpy.ndarray): Input NumPy array.
        smin (float): Minimum value of the range.
        smax (float): Maximum value of the range.

    Returns:
        numpy.ndarray: An array containing the indices of elements in the specified range.
    """
    # Create a boolean mask for elements within the range
    mask = (s > smin) & (s < smax)

    # Use np.where to get the indices where the mask is True
    indices_within_range = np.where(mask)[0]

    return indices_within_range


def dynamic_spectrum_plotter(dyn, t, freq, xlabel, ylabel, **kwargs):
    '''
    Plot the dynamic spectrum dyn as a function of t (be in time or position) 
    on the channels freq (be it frequency or wavelength) given the x/ylabels
    dyn -- dynamic spectrum as numpy matrix
    t -- time as numpy array
    freq -- frequency as numpy array
    xlabel/ylabel -- string with axis labels
    '''
    kwargs.setdefault("vmin", 0)
    kwargs.setdefault("aspect", "auto")
    kwargs.setdefault("interpolation", "none")
    plt.imshow(dyn,
           origin='lower',
           extent=ext_find(t, freq), **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
def secondary_spectrum_plotter_time( fd, tau, SS, per_low, per_high , xlimit, xlabel, ylabel):
    '''
    Plot the secondary spectrum from time (as opposed to the secondary spectrum from position)
    fd is the doppler shift
    tau is the geometric delay
    SS is the secondary spectrum
    per_low is the low percentile for logarithmic scaling for plotting
    per_high is the high percentile for logarithmic scaling for plotting
    xlimit is the symmetric x limit for plotting
    '''
    plt.imshow(SS, 
           norm=LogNorm(vmin=np.percentile(SS , per_low),vmax=np.percentile(SS , per_high)), 
           origin='lower',
           aspect='auto', 
           extent=ext_find( fd , tau ))
    
    plt.xlim((- xlimit , xlimit))
    plt.ylim(0,None)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.colorbar()
    
def secondary_spectrum_plotter_time2( fd, tau, SS, vmin, vmax , xlimit, xlabel, ylabel):
    '''
    Plot the secondary spectrum from time (as opposed to the secondary spectrum from position)
    fd is the doppler shift
    tau is the geometric delay
    SS is the secondary spectrum
    per_low is the low percentile for logarithmic scaling for plotting
    per_high is the high percentile for logarithmic scaling for plotting
    xlimit is the symmetric x limit for plotting
    '''
    plt.imshow(SS, 
           norm=LogNorm(vmin=vmin,vmax=vmax), 
           origin='lower',
           aspect='auto', 
           extent=ext_find( fd , tau ))
    
    plt.xlim((- xlimit , xlimit))
    plt.ylim(0,None)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.colorbar()
    

def secondary_spectrum_noSS( ref_array , freq, dyn, space, normalization):
    '''
    Calculate the 2d FFT of the dynamic spectrum from ref_array input (either time or position on the screen)
    the frequency band freq
    and dynamic spectrum dyn
    space just indicates if ref_array is time or position along the screen
    
    This function retuns
    doppler shift fd
    geometric delay tau
    dynamic spectrum Fourier Transform dyn_FT
    '''
    
    if (space == 'time'):
        #computing the doppler shift
        fd = fft_axis(ref_array , u.mHz)
    elif (space == 'position'):
        fd = fft_axis(ref_array , u.Mm**(-1))
    #computing the geometric delay
    tau = fft_axis(freq, u.us)
    #computing the 2D fourier transform
    dyn_FT = np.fft.fftshift(np.fft.fft2(dyn, norm = normalization))
    
    return fd, tau, dyn_FT    
    
def generate_n_minus_1_x_2_array(idx):
    """
    Generate a (n-1)x2 NumPy array based on the specified pattern.

    Parameters:
        idx (numpy.ndarray): Input NumPy array of length n.

    Returns:
        numpy.ndarray: A (n-1)x2 NumPy array following the specified pattern.
    """
    if len(idx) < 2:
        raise ValueError("Input array 'idx' must have at least 2 elements.")

    # Initialize the result array with zeros
    result = np.zeros((len(idx) - 1, 2), dtype=np.int32)

    for i in range(len(idx) - 1):
        result[i, 0] = idx[i] 
        result[i, 1] = idx[i+1]

    return result

def interpolate_and_evaluate(x0, SS, xlims, xint_range, N, kind):
    """
    Interpolate the matrix SS along its columns and evaluate on np.linspace(-xint_range, xint_range, N).

    Parameters:
    - x: numpy.ndarray, array of x values
    - SS: numpy.ndarray, input matrix
    - xlims: float, maximum value for the range [-xlims, xlims] for which to interpolate SS
    - xint_range: float, range for interpolation evaluation [-xint_range, xint_range]
    - N: int, number of points for evaluation of x0 (has to be odd)
    - kind: string, interpolation kind

    Returns:
    - interpolated_evaluated_matrix: numpy.ndarray, interpolated and evaluated matrix
    """
    # Check if N is positive and odd
    
    if N < 1 or N % 2 == 0:
        raise ValueError("N should be a positive odd integer")
        
    #find upper indeces to interpolate over  
    indices = find_indices_inside_range( [-xlims, xlims] , x0.value)
    x = x0[indices]

    
    #getting the unit
    xunit = x0.unit
    

    # Create an array of x values for interpolation evaluation
    x_interp = np.linspace(-xint_range, xint_range, N) 

    # Initialize an empty matrix for interpolated and evaluated values
    interpolated_evaluated_matrix = np.zeros((SS.shape[0], N))

    # Interpolate and evaluate each row of SS
    for i in range(SS.shape[0]):
        # Create an interpolation function for the i-th row of SS
        interp_func = interp1d(x, SS[i, indices], kind=kind, fill_value='extrapolate')

        # Evaluate the interpolation function on the x_interp values
        interpolated_evaluated_matrix[i, :] = interp_func(x_interp)

    return interpolated_evaluated_matrix, x_interp * xunit
    
def d2str(a, n):
    # Format the number with n decimal points
    result = "{:.{}f}".format(a, n)
    return result


def find_indices_resampling(x0, xmin, xmax, sign):
    """
    Find and return the indices of values of x inside the range (xmin, xmax).

    Parameters:
    - x: NumPy array, input array
    - xmin: float, minimum value of the range
    - xmax: float, maximum value of the range

    Returns:
    - NumPy array, indices of values inside the range (xmin, xmax)
    """
    x = x0[1:]
    if sign > 0:
        # Regular ascending range
        return np.where((x >= xmin) & (x < xmax))[0]+1
    else:
        # Descending range (xmin is a maximum, and xmax is a minimum)
        return np.where((x < xmin) & (x >= xmax))[0]+1


def resampler( lensPos, dyn, ds):
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
    
    
    #create new uniform array if position is increasing
    if (sign_val > 0):
        position_res = np.arange(position_reg[0], position_reg[-1], ds    )
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

def find_similar_regions( pos1, pos2 ):
    """
    Function to find the indeces of the regions that
    pos1 and pos2 have in common
    """
    
    #get the ranges that each array spans
    range1 = np.array([pos1[0], pos1[-1]])
    range2 = np.array([pos2[0], pos2[-1]])
    
    #find indices of the position that are inside the other's region
    indices1 = find_indices_inside_range(range2, pos1)
    indices2 = find_indices_inside_range(range1, pos2)
    
    #if the two indices dont coincide in length simply remove 
    #the element with the biggest amount of separation
    #between the arrays
    
    if len(indices1) < len(indices2):
        
        diff1 = abs( pos1[indices1][0] - pos2[indices2][0] )
        diff2 = abs( pos1[indices1][-1] - pos2[indices2][-1] )
        
        if diff1 > diff2 : 
            
            return indices1, indices2[1:]
        else:
            return indices1, indices2[:-1]
        
        
    elif len(indices1) > len(indices2):
        
        diff1 = abs( pos1[indices1][0] - pos2[indices2][0] )
        diff2 = abs( pos1[indices1][-1] - pos2[indices2][-1] )
        
        if diff1 > diff2 : 
            
            return indices1[1:], indices2
        else:
            return indices1[:-1], indices2
        
    else:
        
        return indices1, indices2
    
    
def find_zero_crossings_indices(y):
    # Find indices where the sign of consecutive elements changes
    sign_changes = np.where(np.diff(np.sign(y)))[0]

    # Find the indices closest to zero| by comparing the absolute values
    zero_crossings_indices = []
    for i in sign_changes:
        if i == 0:
            zero_crossings_indices.append(i)
        else:
            index_before = i - 1
            index_after = i

            # Choose the index with the absolute value closest to zero
            if abs(y[index_before]) < abs(y[index_after]):
                zero_crossings_indices.append(index_before)
            else:
                zero_crossings_indices.append(index_after)

    return zero_crossings_indices


def Ad_projection_unitless(t, nu, phase, A, delta):
    """
    t has to be unitless in hour, 
    phase in radian
    nu in radian
    Function to use the A, delta coefficients from the 
    projected position:
    r(t) = A Omega_A t + f(v) cos(theta + delta) 
    with theta being the orbital phase
    
    """
    #orbital period
    Pb = 0.10225156248 * u.day
    factor = (1 / Pb.to(u.hour)).value
    
    ##Eccentricity
    ecc=0.0877775
    
    # f(v)
    fv = (1 - ecc**2) / (1 + ecc* np.cos(nu))
    
    #cosine and sine parts of cos(theta + delta)
    cosx = fv * np.cos(phase) * np.cos(delta ) 
    sinx = fv * np.sin(phase) * np.sin(delta ) 
    
    
    proj = ( A * factor * t + (cosx - sinx) )
    
    return proj - proj[0]

def Ad_overlap(peaks, peak_widths, t, nu, phase, A, delta):
    """
    Function to compute if the given value of delta is permited for a given 
    value of A given the roots of velocity must align with peaks
    peaks indicate the locations of the turning points 
    and peak_widths indicate their range of validity
    """
    pos = Ad_projection_unitless(t = t, nu = nu, phase = phase, A = A, delta = delta)
    velo = np.gradient( pos, t )
    zeros0 = find_zero_crossings_indices(velo)
    
    if len(zeros0) == len(peaks):
    
        logic_values = np.abs( t[zeros0] - peaks ) <= peak_widths
        
        return np.all(logic_values)
    else:
        return False
    
def Ad_overlap_finder(peaks, peak_widths, t, nu, phase, A, crange):
    
    cmin = 0
    cmax = 0
    
    cstorage = []
    
    for i in range(len(crange)):
        if Ad_overlap(peaks = peaks, peak_widths = peak_widths, t = t, nu = nu, phase = phase, A = A, delta = crange[i]) == True:
            cstorage += [crange[i]]
        
    if len(cstorage) > 1:
        cmin = np.min( np.array(cstorage))
        cmax = np.max( np.array(cstorage))
    
    return (cmin, cmax)

def param_space_array(Astor, dstor, spacing):
    """
    Function to get equally distributed points in the sample space of allowed paramters between A and delta
    the storage arrays for A and delta are given as Astor and dstor, as well as the desired spacing. 
    Function returns a list of similar size to Astor, that has the numpy arrays of delta values 
    that will be sampled over for each A value
    """
    ps_arr = []
    
    for i in range(len(Astor)):
        
        phasetmp = []
        
        n_integers = int(np.abs(dstor[i][1] - dstor[i][0]) / spacing) +1
        
        ps_arr += [np.linspace(dstor[i][0], dstor[i][1], n_integers+2)[1:-1]]
        
    return ps_arr

def meerkat_data_extractor( folder_path ):
    """
    Function to extract a list containing all observations given by folders
    1, 2, 3, 4 as taken from meerkat data
    
    takes folder path all the osbervations are taken
    """

    # Function to get folder names that have subfolders 1234 and sort them in those groups
    def get_subfolder_names(folder_path):
        return [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]

    # Root folder path
    root_folder = '/fs/lustre/scratch/montalvo/meerkat/Obs'

    # Initialize lists for each subfolder number
    subfolder_lists = {1: [], 2: [], 3: [], 4: []}

    # Iterate over all folders in the root folder
    for date_folder in os.listdir(root_folder):
        date_folder_path = os.path.join(root_folder, date_folder)

        # Check if the path is a directory
        if os.path.isdir(date_folder_path):
            # Check for the presence of subfolders 1, 2, 3, 4
            subfolders = get_subfolder_names(date_folder_path)
            for subfolder_num in [1, 2, 3, 4]:
                if str(subfolder_num) in subfolders:
                    subfolder_lists[subfolder_num].append(date_folder)

    #function to sort the dates in ascending order
    sorted_1234 = []
    for i in range(len(subfolder_lists)):

        # Convert folder_names to datetime objects
        dates_tmp = [datetime.strptime(date_str, '%Y-%m-%d-%H:%M:%S') for date_str in subfolder_lists[i+1]]

        # Sort the datetime objects
        sorted_dates_tmp = sorted(dates_tmp)

        # Convert sorted datetime objects back to strings
        sorted_1234 += [[date.strftime('%Y-%m-%d-%H:%M:%S') for date in sorted_dates_tmp]]

    ## function to group dates inside the subfolders 1234 into events on a single day
    full_sorted = []
    for i in range(len(sorted_1234)):
        date_groups_tmp = {}

        for folder_name in sorted_1234[i]:
            date = datetime.strptime(folder_name, '%Y-%m-%d-%H:%M:%S')
            date_key = date.strftime('%Y-%-m-%-d')  # Convert date_key to string format 'YYYY-M-D'
            if date_key in date_groups_tmp:
                date_groups_tmp[date_key].append(folder_name)
            else:
                date_groups_tmp[date_key] = [folder_name]

        full_sorted += [date_groups_tmp]
        
    return full_sorted


def meerk_dyn_sticher(full_sorted, band_index, key_obs):
    """
    Function to stitch together several observations and return a time array, dynamic spectrum, and frequency array
    
    
    """
    dyn_path = ('/fs/lustre/scratch/montalvo/meerkat/Obs/' 
       + full_sorted[band_index-1][key_obs][0] 
        + '/' + str(band_index) + '/J0737-3039A_' + full_sorted[band_index-1][key_obs][0] + '_zap.ar.dynspec')

    dyn = Dynspec(
    filename=dyn_path, process=False, verbose = False
    )

    for i in range( 1, len( full_sorted[band_index-1][key_obs] ) ):
        dyn_path = ('/fs/lustre/scratch/montalvo/meerkat/Obs/' 
           + full_sorted[band_index-1][key_obs][i] 
            + '/' + str(band_index) + '/J0737-3039A_' + full_sorted[band_index-1][key_obs][i] + '_zap.ar.dynspec')
        dyn_tmp = Dynspec(
            filename=dyn_path, process=False, verbose = False
            )
        dyn += dyn_tmp
    
    return dyn.times * u.s.to(u.hour) * u.hour, dyn.dyn, dyn.freqs * u.MHz, dyn.mjd


def Ado_projection_unitless(t, nu, phase, A, do):
    """
    t has to be unitless in hour, 
    phase in radian
    nu in radian
    Function to use the A, delta coefficients from the 
    projected position:
    r(t) = A Omega_A t + f(v) cos(theta + delta) 
    with theta being the orbital phase
    
    """
    #orbital period
    Pb = 0.10225156248 * u.day
    factor = (1 / Pb.to(u.hour)).value
    
    ##Eccentricity
    ecc=0.0877775
    
    #orbital inclination
    #Kramer
    ip = 87.82970787 * u.deg
    
    #Rickett
    #ip = 89.35 * u.deg
    #ip = 90.65 * u.deg
    
    
    # f(v), amplitude, and delta
    fv = (1 - ecc**2) / (1 + ecc* np.cos(nu))
    amp = np.sqrt( np.cos(do)**2 + np.sin(do)**2 * np.cos(ip)**2 )
    delta = np.arctan( np.tan(do) * np.cos(ip) )
    
    
    #cosine and sine parts of cos(theta + delta)
    cosx = fv * amp * np.cos(phase) * np.cos(delta ) 
    sinx = fv * amp * np.sin(phase) * np.sin(delta ) 
    
    
    proj = ( A * factor * t + (cosx - sinx) )
    
    return proj - proj[0]

def Ado_overlap(peaks, peak_widths, t, nu, phase, A, do):
    """
    Function to compute if the given value of delta is permited for a given 
    value of A given the roots of velocity must align with peaks
    peaks indicate the locations of the turning points 
    and peak_widths indicate their range of validity
    """
    pos = Ado_projection_unitless(t = t, nu = nu, phase = phase, A = A, do = do)
    velo = np.gradient( pos, t )
    zeros0 = find_zero_crossings_indices(velo)
    
    if len(zeros0) == len(peaks):
    
        logic_values = np.abs( t[zeros0] - peaks ) <= peak_widths
        
        return np.all(logic_values)
    else:
        return False

def Ado_overlap_finder(peaks, peak_widths, t, nu, phase, A, crange):

    cmin = 0
    cmax = 0

    cstorage = []
    
    cunit = crange.unit
    
    crange_u = crange.value

    for i in range(len(crange)):
        if Ado_overlap(peaks = peaks, peak_widths = peak_widths, t = t, nu = nu, phase = phase, A = A, do = crange[i]) == True:
            cstorage += [crange_u[i]]

    if len(cstorage) > 1:
        cmin = np.min( np.array(cstorage))
        cmax = np.max( np.array(cstorage))

    return (cmin, cmax)