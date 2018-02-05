"""
Create daily calibration values for the ceilometers.

Create by Emma Hopkin: March 2016
Edited by Elliott Warren: 05 Feb 2018

Cloud Calibration of Vaisala Ceilometers - Single Day Calibration Coefficient
Based on theory in O'Connor et al. (2004)
"""
import sys
sys.path.append('C:/Users/Elliott/Documents/PhD Reading/LUMO - Sensor network/calibration/utils')

import os 
from os.path import join 
import iris  # data management and plotting library from met office
import ncUtils # library for reading netCDF files
import matplotlib.pyplot as plt 
from netCDF4 import Dataset 
import numpy as np 
from scipy import stats
import copy
from scipy.interpolate import interp1d
#import VaisCeil_Calibration_utils_EH as EH_utils ###
import operator
import time as Time
#########################################################################################
#Note that this will apply to clear sky profiles as well but this will be filtered out later so this is not an issue
def find_cloud(beta_data):
    """
    Find the location of the max beta
    Take cloud range as 15 gates above and below
    return array with 30gate range beta only; others set to nan
    """
    arr = np.copy(beta_data)
    for i in xrange (len(np.transpose(arr))):
       

        index, value = max(enumerate(arr[:,i]), key=operator.itemgetter(1))
        loc_above = index + 15 
        loc_below = index - 15
        arr[loc_above:,i] = np.nan #set everything above to nan
        arr[:loc_below,i] = np.nan #set everything below to nan
    return(arr)
    
#Only applied upto about 2.4km as cloud above this will be discarded in calibration anyway
#Corrections for above this available from source
def scatter_correct_Vais(cloud_beta, range_data):
    """
    Input: raw beta data
    Output: beta with scattering correxction applied
    Apply multiple scattering correction - source:  http://www.met.reading.ac.uk/~swr99ejo/lidar_calibration/index.html
    """
    Scat_correct_b = np.copy(cloud_beta)
    range_km = range_data/1000  
    #Apply height dependent eta to beta values - i.e. multiply out multiple scattering
    #Note these values are instrument dependent
    ind1 = np.where(range_km< 0.250)
    Scat_correct_b[ind1,:]=Scat_correct_b[ind1,:]* 0.82881
    ind2 = np.where((np.abs(range_km - 0.375)) < 0.125)
    Scat_correct_b[ind2,:]=Scat_correct_b[ind2,:]*0.82445
    ind3 = np.where((np.abs(range_km - 0.625)) < 0.125)
    Scat_correct_b[ind3,:]=Scat_correct_b[ind3,:]*0.81752
    ind4 = np.where((np.abs(range_km - 0.875)) < 0.125)
    Scat_correct_b[ind4,:]=Scat_correct_b[ind4,:]*0.81021
    ind5 = np.where((np.abs(range_km - 1.125)) < 0.125)
    Scat_correct_b[ind5,:]=Scat_correct_b[ind5,:]*0.80241	
    ind6 = np.where((np.abs(range_km - 1.375)) < 0.125)
    Scat_correct_b[ind6,:]=Scat_correct_b[ind6,:]*0.79356
    ind7 = np.where((np.abs(range_km - 1.625)) < 0.125)
    Scat_correct_b[ind7,:]=Scat_correct_b[ind7,:]*0.78595
    ind8 = np.where((np.abs(range_km - 1.875)) < 0.125)
    Scat_correct_b[ind8,:]=Scat_correct_b[ind8,:]*0.77877 
    return (Scat_correct_b)    


#Merge scattering correction with data in beta cube
def corr_beta (Scat_beta, beta_data):
    """
    Locate nans placed by finding cloud(see above)
    replace with beta values
    s_cor_beta is array of beta with scatting correction applied to 30 gates around max beta
    """
    s_cor_beta = np.copy(Scat_beta)

    for prof in  xrange (len(np.transpose(s_cor_beta))):
        index_n = np.isnan(Scat_beta[:,prof])
        thenans = np.where(index_n == True)
        for locnan in xrange(len(thenans)):
           s_cor_beta[(thenans[locnan]), prof] = beta_data [(thenans[locnan]), prof]
       
       
    return(s_cor_beta)  
    
def lidar_ratio(Scat_correct_b,range_data, Instrument, Lower_Gate, Upper_Gate, range_resol):
    """
    Input:scatter corrected beta data (attenuated backscatter coefficient)
    Output: lidar ratio (ratio backscatter coefficient to extinction coefficient)
    """
    # Gate size dependent on instrument
    # Gate2400 marks top of integration - must be 2.4km for H2_off instruments (currently true for Met Office). Could be increased for others
    #begin marks start of integration - avoid problematic gates and worst of overlap in CL51s
   
    # Integrated from 0.2 - 2400km 
    inc_beta = Scat_correct_b[Lower_Gate:Upper_Gate,:]      #betas between 0.2 - 2400km
    integrate_S = ((sum(inc_beta))*range_resol)
    S = ((integrate_S*2)**-1)
    
    return(S) 
    
###########################################################################
#     ######   Calibration with BL transmission correction  ##########
###########################################################################
def transmittance(aer_S, B_prof):
    """
    Calculate a correction due to attenuation of the beam by aerosol in the boundary layer (two way)
    aer_S is an assumed lidar ratio for the aerosol - typically 20-80 sr
    """
    #print 'S', aer_S
    #print B_prof
    ex_prof = aer_S * B_prof
    #print ex_prof
    trans2 = (np.exp(-2*ex_prof))
    return (trans2)
    
 #---------------------------------------------------------------------------------------#    
def lidar_ratio_trans_corrected(Scat_correct_b,range_data, Instrument, Lower_Gate, Upper_Gate, range_resol,aer_S):
    """
    Input:scatter corrected beta data (attenuated backscatter coefficient)
    Output: lidar ratio (ratio backscatter coefficient to extinction coefficient)
    """
    # Gate size dependent on instrument
    # Gate2400 marks top of integration - must be 2.4km for H2_off instruments (currently true for Met Office). Could be increased for others
    #begin marks start of integration - avoid problematic gates and worst of overlap in CL51s
   
    # Integrated from 0.2 - 2400km 
    inc_beta = Scat_correct_b[Lower_Gate:Upper_Gate,:]      #betas between 0.2 - 2400km
    
    S = [] 
    B_T2 = []   
    
    for i in xrange(len(np.transpose(inc_beta))):
        peak, value = max(enumerate(Scat_correct_b[:,i]), key=operator.itemgetter(1))#find peak of backscatter (cloud)

        B_aer_prof = ((sum(Scat_correct_b[:(peak-5),i]))*range_resol)#integrate upto just below cloud (aerosol)

        integrate_S = ((sum(inc_beta[:,i]))*range_resol)#integral for S (excludes problem gates below 500m)

        aer_transmittance = transmittance(aer_S,B_aer_prof)#calculate transmission correction for assumed aerosol lidar ratiousing all of profile below cloud

        integrate_S2 = integrate_S / aer_transmittance #correct for BL attenuation with assumed lidar ratio

        prof_S = ((integrate_S2*2)**-1)
        S.append(prof_S)
        B_T2.append(integrate_S2)
    S = np.asarray(S)    
    return(S, B_T2)    
#---------------------------------------------------------------------------------------#  
    
    
 ###########################################################################
#           ######   Instrument Checks   ##########
###########################################################################      
#filter by window
def filter_window(beta_data, window_transmission, window_crit):
    win_tau = np.asarray(window_transmission)
    beta_win = np.copy(beta_data)
    for i in xrange (len(np.transpose(beta_data))):
        if win_tau[i] < window_crit:
            #print '**Window Transmission too low**'
            beta_win[:,i] = np.nan
        else:
            pass
    return(beta_win)
    
 #filter by power
def filter_power(beta_data, laser_pulse, power_crit):
    pulse_energy = np.asarray(laser_pulse)
    beta_pul = np.copy(beta_data)
    for i in xrange (len(np.transpose(beta_data))):
        if pulse_energy[i] < power_crit:
            #print '**Pulse Energy too low**'
            beta_pul[:,i] = np.nan
        else:
            pass
    return(beta_pul)     
    
###########################################################################
#           ######   Attenuated Beta Filters   ##########
###########################################################################
#include only profiles where beta 300m above max is at least 20 times smaller
def filter_300 (beta_data, Instrument, range_resol ):
    """
    Hogan et al (2003b); O'Connor et al (2004) - to be a thick highly attenuating cloud, peak att_Beta must be a factor of 20 times greater than 300m above
    Instrument dependent - different gate sizes
    Where the condition is not passed, value is replaced with a NaN
    """
    # Gate size dependent on instrument
    # Ensure cloud is above 100m
    Gate300m = int(np.round(300/range_resol))
        
    beta_300f = np.copy(beta_data)
    a = np.zeros(Gate300m)
    a = np.transpose(a)
    ex2 = 0
    import operator
    
    for i in xrange (len(np.transpose(beta_300f))):
        profs_sort = np.append(beta_300f[:,i],a)

        index, value = max(enumerate(beta_300f[:,i]), key=operator.itemgetter(1))
        loc_compare_p = index + Gate300m
        compare_p = profs_sort[(loc_compare_p)]
        
        if loc_compare_p < 0 or compare_p*20 > value: 
            beta_300f[:,i] = np.nan
            ex2 = ex2 +1
        else:
            pass

    #print'300m above filtered out: ', ex2 
    
    return (beta_300f)    
#------------------------------------------------------------------------#
#include only profiles where beta 300m below max is at least 20 times smaller
def filter_300below (beta_data,Instrument, range_resol):
    """
    Follows reasoning of 300m above filtering
    Instrument dependent - different gate sizes
    Where the condition is not passed, value is replaced with a NaN
    """
    # Gate size dependent on instrument
    # Ensure cloud is above 100m
    Gate300m = int(np.round(300/range_resol))
        
    beta_300belf = np.copy(beta_data)
    #add on 300m of zeros so bottom of profile isnt compared to top
    a = np.zeros(Gate300m)
    a = np.transpose(a)
    ex2 = 0
    import operator
    
    for i in xrange (len(np.transpose(beta_300belf))):
        profs_sort = np.append(beta_300belf[:,i],a)

        index, value = max(enumerate(beta_300belf[:,i]), key=operator.itemgetter(1))
        loc_compare_p = index - Gate300m
        compare_p = profs_sort[(loc_compare_p)]
        
        if loc_compare_p < 0 or compare_p*20 > value: 
            beta_300belf[:,i] = np.nan
            ex2 = ex2 +1
        else:
            pass
    #print '300m below filtered out: ', ex2 
    
    return (beta_300belf)     
    
 #------------------------------------------------------------------------# 
def total_beta(beta_data, Instrument, range_resol, GateMax):
    """
    Input: scatter corrected beta data from cube, (between selected indices??)
    Output: integrated beta for each 30s profile
    """

    inc_beta = beta_data[5:GateMax,:]   #betas above first 5 gates (near view issues)
    integrate_beta = ((sum(inc_beta))*range_resol)
    return (integrate_beta)


def inteB_belowCBH(beta_data,range_data, Instrument, range_resol):
    """
    Integrate beta up to 100m below max beta 
    ???Where CBH is below 200m range, beta is returned as zero - changed to nan
    """
    G100m = np.round(100/range_resol)
     
    # Sum beta_data values of profile
    inte_beta = []
    a = len(np.transpose(beta_data))
    for prof in xrange(a):
        loc_belowCBH = np.where(beta_data[:,prof] == np.max(beta_data[:,prof]))[0][0]
        loc = loc_belowCBH - G100m #100m below

        inc_beta = beta_data[5:loc,prof]      #betas between gate 5 - 100m below
        integrate_beta = ((sum(inc_beta))*range_resol)
        inte_beta.append(integrate_beta)
    inte_beta[inte_beta==0] = np.nan
    return(inte_beta)   
#---------------------------------------------------------------------------------------#
def filter_ratio(beta_data,range_data, ratio_val, Instrument, range_resol,Cal_maxheight):
    """
    Hogan et al (2003b); O'Connor et al (2004)    - exclude strong background aerosol events 
    Input: raw beta data, set required ratio
    integrate over full profile values and integrate up to 100m below CBH
    If beta below CBH represents more than 5% of total, profile discarded
    Output: beta with nans for profiles that don't qualify
    """
    integrated_B = total_beta(beta_data, Instrument, range_resol, Cal_maxheight)
    integrated_belCBH_B = inteB_belowCBH(beta_data,range_data, Instrument, range_resol)
    B_ratio = integrated_belCBH_B/integrated_B
    #B_ratio = beta_ratio.tolist()
    ratiof_B = np.copy(beta_data)
    filt_out = 0
    #eliminate profiles where beta below CBH represents more than 5% of total
    for i in xrange(len(B_ratio)):
        if B_ratio[i] > ratio_val:
            ratiof_B[:,i] = np.nan
            filt_out = filt_out +1
        elif B_ratio[i] < 0:
            ratiof_B[:,i] = np.nan 
            filt_out = filt_out +1       
        else:
            pass 
    #print 'ratio filtered out: ', filt_out   
    return(ratiof_B, B_ratio)    
    
 ############### Bring att_beta filtering together######################
def locate_nan(arr):
    """
    Input: version of beta with nans created by a filter
    Output: location of nans in beta array
    """
    index_n = np.isnan(arr)                 #e.g. beta_300f
    find_n = np.where(index_n == True)
    loc_n1 = find_n[1]
    loc_n2 = np.squeeze(loc_n1)
    loc_n = np.unique(loc_n2)
    return (loc_n)
#---------------------------------------------------------------------------------------#
def filter_bynan (loc_n,S):
    """
    Input: Output for locate_nan() and second array to filter (usually S)
    Output: Filtered version of e.g. S (nans in place)
    """
    #print np.shape(S)
    filt_S = np.copy(S)
    for loc in xrange(len(loc_n)):
        filt_S[(loc_n[loc])] = np.nan
    return (filt_S)
#---------------------------------------------------------------------------------------#
def step1_filter(beta_data, range_data,ratio_val,S, Instrument, range_resol):
    """
    Using scatter corrected data, run through several filters to remove profiles unsuitable for calibration
    filter1: aerosol from 100m below CBH (max B) to gate5 must not be more than (e.g).5% of total integrated B (0.2-2.4km)    
    filter2: B 300m above max B must be 20x smaller
    filter3: B 300m below max B must be 20x smaller 
    filter4: Log Max B must be at least -5 m-1 sr-1
    NB. filter functions above, called by this one
    beta data with nans for profiles that did not pass the filters
    OUTPUT: S with nan where filters not passed
    """
    filter1, B_ratio = filter_ratio(beta_data, range_data, ratio_val, Instrument, range_resol, Cal_maxheight)    
    filter2 = filter_300 (filter1, Instrument, range_resol)
    filter3 = filter_300below(filter2, Instrument, range_resol)    
       
    loc_beta_f = locate_nan(filter3)
    filtered_S = filter_bynan(loc_beta_f, S)
    return(filtered_S)    
    
#---------------------------------------------------------------------------------------#      
###########################################################################
#               ######   Lidar Ratio Filter   ##########
###########################################################################
def step2_Sfilt (S, S_range_percent, no_of_profs):
    """
    New version of step2_Sfilt which requires 7 profiles in a row to be
    within 10% of the mean of the 7 profiles. Steps forward one profile 
    at a time. If the profiles meet the criteria, all 7 are copied, else they
    are left as zeros. All remaining zeros converted to nans
    """
    plus = 100. + S_range_percent
    minus = 100. - S_range_percent
    const_S = np.zeros(len(S))
    
    for i in xrange(len(S)-(no_of_profs - 1)):
        group = S[i:i+(no_of_profs)]
        
        mean_of_group = np.mean(group)
        range_check = group/mean_of_group*100
        if (np.min(range_check) > minus and np.max(range_check) < plus):
            const_S[i:i+(no_of_profs)] = S[i:i+(no_of_profs)]
        else:
            pass
            

    const_S[const_S==0.] = np.nan

    return(const_S)

#---------------------------------------------------------------------------------------#
def Plot_ashist(variable):
    v = np.copy(variable)
    v[np.isnan(v)] = 0
    vartoplot = v[np.nonzero(v)]
    if len(vartoplot) > 1:
        #b = np.round((np.max(vartoplot)))
        b=1
        #plt.figure(figsize = (6,4))
        counts, bins, range = plt.hist(vartoplot, bins = (10*b), range = (0,b))
        #plt.xticks(np.arange(0, 110, 10.0), fontsize = 18)
        plt.yticks(fontsize = 18)
        #plt.axis(xmax = b+1, xmin = 16)
        bin_centers = 0.5 * np.diff(bins) + bins[:-1]
        
        plt.title('S',fontsize = 20)
        plt.xlabel('Apparent S [sr]', fontsize = 18)
        plt.ylabel('Frequency', fontsize = 18)
        plt.tight_layout()
        
        #plt.axis(xmax = 100)
        #plt.show()
    else:
        counts = 0    
    return(counts,len(vartoplot)) 



def S_Stats(Step2_S, Cal_hist):
    """
    Calculate daily mode for S
    if number of profiles in mode is less than 10 (5 mins), daily stats will not be used in the calibration (ie are set to nan) - this is derived from Cal_hist
    Cal_hist gives histogram values so max gives the mode
    """
    var = np.copy(Step2_S)
    peak = np.max(Cal_hist)
    if peak > 10.:
        print '@@@@@@@@ No. of profiles in peak =', peak, '@@@@@@@'
        m = np.asarray(var)
        
        m[np.isnan(m)]= 0
        m = m[np.nonzero(m)]
        mean = np.mean(m)           ###mean   
        print 'calibration mean = ', mean
        
        stdev = np.std(m)
        print 'std deviation = ', stdev        
        
        sem = stats.sem(m)
        print 'std error = ', sem
        
        m2 = np.round((m.tolist()),1)
        mode_arr = stats.mode(m2)
        mode = mode_arr[0][0]
        print 'calibration mode = ', mode
        
        median = np.median(m)
        print 'calibration median = ', median
    else:
        print '###### No. of profiles in peak = ', peak, '#####'
        mode = np.nan
        mean = np.nan
        median = np.nan
        sem = np.nan
        stdev = np.nan
    return (peak,mode, mean, median, sem, stdev)
########################################################################################################################################
"""
Select files for use in calibration
Met Office files are hourly - ie should be using 24 files for each daily calibration
""" 
def user_input_variables():
    year_to_open = str(raw_input ('year of file (format YYYY): '))
    month_to_open = str(raw_input ('month of file (format MM): '))
    day_to_open = str(raw_input ('day of file (format DD): '))
    date = str(year_to_open + month_to_open + day_to_open)
    WMO = str(raw_input ('WMO Number (format #####): '))
    return(WMO,date)

def get_hourly_files(path,WMO,date):
    list_of_saved_files = "ls_out"
    filepath = path+'/*'+WMO+'*'+date+'*'
    command = "ls "+filepath+" > "+join(path,list_of_saved_files)
    os.system(command)
    if os.path.isdir(path):
        fn = open(join(path,list_of_saved_files))
        lines = [line.strip() for line in fn]
        fn.close()    
    else:
        print 'ERROR: Directory does not exist'
        lines = []
    return (lines)

# -----------------------------------------------------
# Setup
# -----------------------------------------------------

#Arrays to store variables from each file
time_data = []
start_time = []
end_time = []
window_transmission = []
laser_energy = []
laser_temp = []
CL = []
upper_CL = []
lower_CL = []

#Define Calibration Variables
Instrument = 'cl31'
Cal_minheight = 200. #m Note 200-500m tends to be noiser but mode is usually consistent with other ranges
Cal_maxheight = 2000. #m
ratio_filt = 0.05 #between 0.05-0.10 works best - increasing to 0.1 allows more profiles to be used but a wider distribution of calibration
no_of_profs = 5 #No of consecutive profiles required. Must be odd integer. Recommend 3/5/7/9

# Get list of hourly (ceilometer?) files using ls command
path = '/net/glusterfs/scenario/users/qt013194/Data/L1_FILES/L1_FILES'
WMO,date = user_input_variables()
filenames = get_hourly_files(path,WMO,date)

#check for 24 files (ie one day's worth) - print warning and continue if less 
if len(filenames) < 23:
    print 'WARNING: less files than expected - not 24 hours of data'


#########################################################################################
# Get Common Variables
path_to_file = join(path, filenames[0])
print "File to open and read ",path_to_file 
list_of_contents=ncUtils.nc_listatt(path_to_file)
print ' Attributes : ', list_of_contents

site = ncUtils.nc_getatt(path_to_file, 'site_location')
inst_title = ncUtils.nc_getatt(path_to_file,'title')
print inst_title

list_variables=ncUtils.nc_listvar(path_to_file)
obsfile=Dataset(path_to_file)

#Read variables from L1 file
#Wavelength
wavelength = obsfile.variables['l0_wavelength'][:]

#Get file dependant variables
for ll in filenames:
    path_to_file = join(path, ll)
    list_of_contents=ncUtils.nc_listatt(path_to_file)
    list_variables=ncUtils.nc_listvar(path_to_file)
    obsfile=Dataset(path_to_file)
    #Time variables
    time = obsfile.variables['time']
    file_time_data = obsfile.variables['time'][:]
    time_data = np.concatenate((time_data, file_time_data))
    time_resol = obsfile.variables['time_resol'][:]
    file_start_time = Time.strftime("%Y-%m-%d %H:%M:%S", Time.gmtime(time_data[0]*86400))
    start_time.append(file_start_time)
    file_end_time = Time.strftime("%Y-%m-%d %H:%M:%S", Time.gmtime(time_data[-1]*86400))
    end_time.append(file_end_time)
    
    #Range Variables
    range = obsfile.variables['range']
    range_data = obsfile.variables['range'][:]
    range_resol = obsfile.variables['range_resol'][:]

    #CLoud Base Height (3 layers)
    cbh = obsfile.variables['cbh'][:]

    #Additional Variables for Calibration Suitability Checks
    file_window_transmission = obsfile.variables['window_transmission']
    window_transmission = np.concatenate((window_transmission, file_window_transmission))
    file_laser_energy = obsfile.variables['laser_energy']
    laser_energy = np.concatenate((laser_energy, file_laser_energy))
    file_laser_temp = obsfile.variables['laser_temp']
    laser_temp = np.concatenate((laser_temp, file_laser_temp))

    rcs= obsfile.variables['rcs_0'][:]     
    #Convert from 100000 km-1 sr-1 to m-1 sr-1
    beta_data =  np.transpose(rcs*(1e-8))     

    #Define Instrument dependent variables
    Lower_Gate = Cal_minheight/range_resol
    Upper_Gate = Cal_maxheight/range_resol


    ###CALIBRATION###
    #Multiple Scattering Correction 
    cloud_beta = find_cloud(beta_data)
    Scat_correct_beta = scatter_correct_Vais(cloud_beta, range_data)
    beta_arr = corr_beta (Scat_correct_beta, beta_data)


    #run checks on instrument transmission
    beta_arr = filter_window(beta_arr,file_window_transmission , 90.)
    beta_arr = filter_power(beta_arr, file_laser_energy, 90.)


    #Calculate Apparent Lidar Ratio for each profile
    #S = lidar_ratio(beta_arr,range_data, Instrument, Lower_Gate, Upper_Gate, range_resol)
    #####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
          #Calculate Apparent Lidar Ratio for each profile
          ##For calibration without a BL transmission correction use S = lidar_ratio and section commented out below
    #S = lidar_ratio(beta_arr,range_data, Instrument, Lower_Gate, Upper_Gate, range_resol)
    C_withdiff_T2 = []
    S_in_aerosol = [20.,25.,30.,35.,40.,50.,60.,70.]
    for Ss in S_in_aerosol:
       S_withT2, B_T2 = lidar_ratio_trans_corrected(beta_arr,range_data, Instrument, Lower_Gate, Upper_Gate, range_resol, Ss)
       if np.isnan(S_withT2).any():
          print 'Found some values in Apparent Lidar ratio set as NaN - so unable to continue processing due to differences in array sizes'
       else: 
          #Apply filters to ensure only liquid water cloud profiles used   
          Step1_S_withT2 = step1_filter(beta_arr, range_data,ratio_filt,S_withT2, Instrument, range_resol)
          Step2_S_withT2 = step2_Sfilt (Step1_S_withT2, 10, no_of_profs)
          #Convert to calibration coefficient
          C_withT2 = Step2_S_withT2/18.8
          C_withdiff_T2.append(C_withT2)
    C_means = []
    C_stdevs = []    
    C_arr = np.asarray(C_withdiff_T2)    
    for i in xrange(len(np.transpose(C_arr))):
       mean_C_T2 = np.mean(C_arr[:,i])
       C_means.append(mean_C_T2)
       stdev_C_T2 = np.std(C_arr[:,i])
       C_stdevs.append(stdev_C_T2)
       
    #Convert into calibration factor
    file_CL = (1./np.asarray(C_means))  #Should this be converted back from m-1 sr-1 ie divide by (1e-8)????


    CL = np.concatenate((CL, file_CL))

    #Upper limit due to aerosol lidar ratio
    file_upper_CL = (1./C_withdiff_T2[-1])
    upper_CL = np.concatenate((upper_CL,file_upper_CL))
    #Lower limit due to aerosol lidar ratio
    file_lower_CL = (1./C_withdiff_T2[0])
    lower_CL = np.concatenate((lower_CL,file_lower_CL)) 

    #####@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@      
          
    #Apply filters to ensure only liquid water cloud profiles used
    #Step1_S = step1_filter(beta_arr, range_data,ratio_filt,S, Instrument, range_resol)  
    #Step2_S = step2_Sfilt (Step1_S, 10, no_of_profs) 

    #Convert to calibration coefficient
    #Cal_C = Step2_S/18.8

    #Convert into calibration factor
    #file_CL = (1./Cal_C)  #Should this be converted back from m-1 sr-1 ie divide by (1e-8)????
    #CL = np.concatenate((CL, file_CL))

#Calibration statistics for mean transmission correction
Cal_hist,mode_no_of_profs = Plot_ashist(CL) 
no_in_peak, day_mode, day_mean, day_median, day_sem, day_stdev = S_Stats(CL, Cal_hist)

#Calibration statistics for upper limit transmission correction
upper_Cal_hist,upper_mode_no_of_profs = Plot_ashist(upper_CL) 
upper_no_in_peak, upper_day_mode, upper_day_mean, upper_day_median, upper_day_sem, upper_day_stdev = S_Stats(upper_CL, upper_Cal_hist)  

#Calibration statistics for lower limit transmission correction
lower_Cal_hist,lower_mode_no_of_profs = Plot_ashist(lower_CL) 
lower_no_in_peak, lower_day_mode, lower_day_mean, lower_day_median, lower_day_sem, lower_day_stdev = S_Stats(lower_CL, lower_Cal_hist)     
################
#Save to NetCDF file

output_file = 'ALC_CloudCal_'+ WMO +'_'+ Instrument + '.nc'

if not(os.path.isfile(output_file)):
    print 'Creating NetCDF File'
    
    ncID = Dataset(output_file, 'w', format = 'NETCDF4')
    
    nc_time = ncID.createDimension('time', None)
    
    
    #Create Variables
    nc_start_time = ncID.createVariable('start_time', 'f8', ('time',))
    nc_end_time= ncID.createVariable('end_time', 'f8', ('time',))
    
    nc_mode_cal_factor = ncID.createVariable('mode_cal_factor', 'f8', ('time',))
    nc_median_cal_factor = ncID.createVariable('median_cal_factor', 'f8', ('time',))
    nc_mean_cal_factor = ncID.createVariable('mean_cal_factor', 'f8', ('time',))
    nc_std_cal_factor = ncID.createVariable('std_cal_factor', 'f8', ('time',))
    
    nc_upper_mode_cal_factor = ncID.createVariable('upper_mode_cal_factor', 'f8', ('time',))
    nc_upper_median_cal_factor = ncID.createVariable('upper_median_cal_factor', 'f8', ('time',))
    nc_upper_mean_cal_factor = ncID.createVariable('mupper_ean_cal_factor', 'f8', ('time',))
    nc_upper_std_cal_factor = ncID.createVariable('upper_std_cal_factor', 'f8', ('time',))

    nc_lower_mode_cal_factor = ncID.createVariable('lower_mode_cal_factor', 'f8', ('time',))
    nc_lower_median_cal_factor = ncID.createVariable('lower_median_cal_factor', 'f8', ('time',))
    nc_lower_mean_cal_factor = ncID.createVariable('lower_mean_cal_factor', 'f8', ('time',))
    nc_lower_std_cal_factor = ncID.createVariable('lower_std_cal_factor', 'f8', ('time',))
                
    nc_calibration_bottom_height = ncID.createVariable('calibration_bottom_height', 'f8', ('time',))
    nc_calibration_top_height = ncID.createVariable('calibration_top_height', 'f8', ('time',))
    nc_laser_wavelength = ncID.createVariable('laser_wavelength', 'f8', ('time',))
    nc_temperature_laser = ncID.createVariable('temperature_laser', 'f8', ('time',))
    nc_energy_laser = ncID.createVariable('energy_laser', 'f8', ('time',))
    nc_window_transmission = ncID.createVariable('window_transmission', 'f8', ('time',))
    
    #Add data
    nc_start_time[0] = time_data[0]
    nc_end_time[0] = time_data[-1]
    nc_mode_cal_factor[0] = day_mode
    nc_median_cal_factor[0] = day_median
    nc_mean_cal_factor[0] = day_mean
    nc_std_cal_factor[0] = day_stdev
    
    nc_upper_mode_cal_factor[0] = upper_day_mode
    nc_upper_median_cal_factor[0] = upper_day_median
    nc_upper_mean_cal_factor[0] = upper_day_mean
    nc_upper_std_cal_factor[0] = upper_day_stdev

    nc_lower_mode_cal_factor[0] = lower_day_mode
    nc_lower_median_cal_factor[0] = lower_day_median
    nc_lower_mean_cal_factor[0] = lower_day_mean
    nc_lower_std_cal_factor[0] = lower_day_stdev 
         
    nc_laser_wavelength[0] = wavelength
    nc_temperature_laser[0] = laser_temp
    nc_energy_laser[0] = laser_energy
    nc_window_transmission[0] = window_transmission
    
else:
    print 'Updating NETCDF File'
    ncID = Dataset(output_file, 'a')
    
    #Add data
    length_nc = ncID.variables['start_time'].size
    ncID.variables['start_time'] [length_nc] = time_data[0]
    ncID.variables['end_time'] [length_nc] = time_data[-1]
    ncID.variables['mode_cal_factor'][length_nc] = day_mode
    ncID.variables['median_cal_factor'][length_nc] = day_median
    ncID.variables['mean_cal_factor'][length_nc] = day_mean
    ncID.variables['std_cal_factor'][length_nc] = day_stdev
    
    nc_start_time = time_data[0]
    nc_end_time = time_data[-1]
    nc_mode_cal_factor = day_mode
    nc_median_cal_factor = day_median
    nc_mean_cal_factor = day_mean
    nc_std_cal_factor = day_stdev
    
    nc_upper_mode_cal_factor = upper_day_mode
    nc_upper_median_cal_factor = upper_day_median
    nc_upper_mean_cal_factor = upper_day_mean
    nc_upper_std_cal_factor = upper_day_stdev

    nc_lower_mode_cal_factor = lower_day_mode
    nc_lower_median_cal_factor = lower_day_median
    nc_lower_mean_cal_factor = lower_day_mean
    nc_lower_std_cal_factor = lower_day_stdev 
    
ncID.close()    













