"""
Calibration for the LUMO ceilometers.

Code create block sets of calibration values based on a date range, which takes into account water vapour absorption.
Saves yearly netCDF files of the outputs in L0 folder. Based on the cumulostratus calibration method developed by
Hopkin et al., in prep. Code designed mainly for CL31 and Lufft ceilometers but adapted here for use solely for LUMO
ceilometers. Functions kept in the LUMO_ceilometer_Utils.py script.

Data needed is:
- L1 BSC files (attenuated backscatter from ceilometers)
- L0 CCW30 files (transmission data)
- Met Office NWP forecast files (water vapour correction: requiring, specific humidity, pressure and temperature)

Created by Elliott Warren Thurs 08/03/2018
Based on (EW_)CloudCal_filt_VaisTemp_LUMO by Emma Hopking, edited by Elliott Warren
"""

import sys
# append dir containing EH's utility library
sys.path.append('C:/Users/Elliott/Documents/PhD Reading/LUMO - Sensor network/calibration/utils')
import EH_Utils as EH
import Cal_Utils as CAL
import Stats_Utils as ST
import LUMO_calibration_Utils as lcu

import os # operating system library to issue Unix commands
from netCDF4 import Dataset # to read standard netCDF files
import numpy as np # numerical python library for arrays and mathematical functions
import datetime as dt
import copy
import ellUtils as eu
import ceilUtils as ceil

# ----------------------------
# Setup
# ----------------------------

# directories
# final datadir structure needs to be...
# /data/its-tier2/micromet/data/2017/London/L1/IMU/DAY/200/
# NOTE: Make sure bsc files are from L0!!
datadir_bsc = 'C:/Users/Elliott/Documents/PhD Reading/LUMO - Sensor network/calibration/data/2015/London/L0/01/'
datadir_ccw30 = 'C:/Users/Elliott/Documents/PhD Reading/LUMO - Sensor network/calibration/data/2015/01/'
datadir_mo ='C:/Users/Elliott/Documents/PhD Reading/LUMO - Sensor network/calibration/data/MO/'

# time to loop through
start_date = dt.datetime(2015, 1, 1)
end_date = dt.datetime(2015, 2, 1)
date_range = eu.date_range(start_date, end_date, 1, 'day')

# site and ceilometer information
site_ins = {'site': 'NK', 'ceil_id': 'CL31-D'}

# settings to tune calibration
ratio_filt = 0.05
maxB_filt = -10  #set high so that this filter is no longer used
cont_profs = 5  #number of continuous profiles required for calibration (must be odd no.)

# ----------------------------
# Define variables
# ----------------------------

profile_total = []  #array of total number of profiles used for calibration for each day
peak_total = []     #array of number of profiles in the mode (from histogram)
modes = []          #array of mode of S for each day
means = []          #array of mean of S for each day
medians = []        #array of median of S for each day
sems = []           #array of standard error of S for each day
stdevs = []         #array of standard deviation of S for each day
C_modes = []
C_medians = []
C_stdevs = []
CL_medians = []
CL_stdevs = []

### 2 where relating to aer correction...
modes2 = []          #array of mode of S for each day
means2 = []          #array of mean of S for each day
medians2 = []        #array of median of S for each day
sems2 = []           #array of standard error of S for each day
stdevs2 = []         #array of standard deviation of S for each day
C_modes2 = []
C_medians2 = []
C_stdevs2 = []
CL_medians2 = []
CL_stdevs2 = []
All_S2 = []

modes_wv = []          #array of mode of S for each day
means_wv = []          #array of mean of S for each day
medians_wv = []        #array of median of S for each day
sems_wv = []           #array of standard error of S for each day
stdevs_wv = []         #array of standard deviation of S for each day
C_modes_wv = []
C_medians_wv = []
C_stdevs_wv = []
CL_medians_wv = []
CL_stdevs_wv = []



# ----------------------------
# Read in data
# ----------------------------

for day in [date_range[17]]: # date_range:

    # Read in L0 backscatter data (need beta, time and range) ---------------

    # Note: L0 heights are corrected for height above ground
    #       L1 heights are NOT corrected and are therefore just the range...
    bsc_filepath = datadir_bsc + 'CL31-A_BSC_KSS45W_'+day.strftime('%Y%j')+'_15sec.nc'

    # read in L0 backscatter data
    bsc_data, _ = ceil.netCDF_read_BSC(bsc_filepath)

    # transpose the backscatter for EH functions
    bsc_data['backscatter'] = np.transpose(bsc_data['backscatter'])

    # create range in [km]
    bsc_data['range_km'] = bsc_data['range'] / 1000.0


    # ------------------------------
    # Apply scattering correction
    # ------------------------------

    # find the cloud based on the max backscatter return, and set the backscatter at all other heights to nan
    cloud_beta = EH.find_cloud(bsc_data['backscatter'])

    # apply the multiple scattering correction for the backscatter that was not the cloud
    Scat_correct_b = lcu.scatter_correct_Vais(cloud_beta, bsc_data['range_km'])

    # apply the multiple scattering to correct the non-cloud backscatter,
    #    and merge the array with the cloud backscatter array
    beta_arr = lcu.corr_beta(Scat_correct_b, bsc_data['backscatter'])

    # ----------------------------------------------
    # Apply water vapour attenuation correction
    # ----------------------------------------------

    # Get full file paths for the day and yesterday's (yest) MO data
    yest_filepath, day_filepath = lcu.mo_create_filenames(day, datadir_mo)

    # if both day's data exist, apply water vapour correction, else set backscatter to nan
    if (os.path.exists(yest_filepath))  & (os.path.exists(day_filepath)):
        # Calculate and apply transmissivity to multiple scattering, corrected backscatter data
        transmission_wv = lcu.mo_read_calc_wv_transmission(yest_filepath, day_filepath, day, bsc_data['range'], bsc_data['time'], bsc_data['backscatter'])
        beta_arr_wv = beta_arr * (1.0 / np.transpose(transmission_wv))
    else:
        beta_arr_wv = beta_arr * np.nan

    # ----------------------------------------------
    # Calculate calibration
    # ----------------------------------------------

    ## 1. S without water vapour correction

    # calculate S, including transmission correction (on non water vapour corrected profiles)
    S, S2 = lcu.lidar_ratio(beta_arr, bsc_data['range_km'])

    # Remove profiles unsuitable for calibration
    Step1_S, profile_B_ratio = lcu.step1_filter(bsc_data['backscatter'], bsc_data['range_km'], maxB_filt, ratio_filt, S)  # aerosol ratio = 5%
    Step1_S2, profile_B_ratio2 = lcu.step1_filter(bsc_data['backscatter'], bsc_data['range_km'], maxB_filt, ratio_filt, S2)  # aerosol ratio = 5%

    ##Apply S Filters
    Step2_S = lcu.step2_Sfilt(Step1_S, 10, cont_profs)  # range in S = 10%
    Step2_S2 = lcu.step2_Sfilt(Step1_S2, 10, cont_profs)
    # remove neg values caused by neg noise
    Step2_S[Step2_S < 0] = np.nan
    Step2_S2[Step2_S2 < 0] = np.nan

    ## 2. S with water vapour correction

    # calculate lidar ratio for the water vapour corrected profiles
    S_wv, S_wv2 = lcu.lidar_ratio(beta_arr_wv, bsc_data['range_km'])

    # filter out bad profiles, unsuitable for calibrations
    Step1_S_wv, profile_B_ratio_wv = lcu.step1_filter(bsc_data['backscatter'], bsc_data['range_km'], maxB_filt, ratio_filt, S_wv)  # aerosol ratio = 5%
    Step2_S_wv = lcu.step2_Sfilt(Step1_S_wv, 10, cont_profs)
    # remove neg values caused by neg noise
    Step2_S_wv[Step2_S_wv < 0] = np.nan

    # -----------------
    # Statistics
    # -----------------

    # Calculate mode and mean
    Cal_hist, no_of_profs = ST.Plot_ashist(Step2_S)  # Histogram of filtered S
    no_in_peak, day_mode, day_mean, day_median, day_sem, day_stdev, dayC_mode, dayC_median, dayC_stdev, dayCL_median, dayCL_stdev = ST.S_mode_mean(
    Step2_S, Cal_hist)

    Cal_hist_wv, no_of_profs_wv = ST.Plot_ashist(Step2_S_wv)  # Histogram of filtered S
    no_in_peak_wv, day_mode_wv, day_mean_wv, day_median_wv, day_sem_wv, day_stdev_wv, dayC_mode_wv, dayC_median_wv, dayC_stdev_wv, dayCL_median_wv, dayCL_stdev_wv = ST.S_mode_mean(
    Step2_S_wv, Cal_hist_wv)

    Cal_hist2, no_of_profs2 = ST.Plot_ashist(Step2_S2)  # Histogram of filtered S
    no_in_peak2, day_mode2, day_mean2, day_median2, day_sem2, day_stdev2, dayC_mode2, dayC_median2, dayC_stdev2, dayCL_median2, dayCL_stdev2 = ST.S_mode_mean(
    Step2_S2, Cal_hist2)


    ## Append statistics for each

    All_S = np.concatenate((All_S, Step2_S))
    All_S2 = np.concatenate((All_S2, Step2_S2))
    # ~~~~
    S_forbox = np.array(Step2_S)
    S_forbox[np.isnan(S_forbox)] = 0
    S_forbox = S_forbox[np.nonzero(S_forbox)]
    if np.max(Cal_hist) > 10:
        S_box.append(S_forbox)
    else:
        S_box.append([0])

    profile_total.append(no_of_profs)
    peak_total.append(no_in_peak)
    modes.append(day_mode)
    means.append(day_mean)
    medians.append(day_median)
    sems.append(day_sem)
    stdevs.append(day_stdev)
    C_modes.append(dayC_mode)
    C_medians.append(dayC_median)
    C_stdevs.append(dayC_stdev)
    CL_medians.append(dayCL_median)
    CL_stdevs.append(dayCL_stdev)

    # profile_total2.append(no_of_profs2)
    # peak_total2.append(no_in_peak2)
    modes2.append(day_mode2)
    means2.append(day_mean2)
    medians2.append(day_median2)
    sems2.append(day_sem2)
    stdevs2.append(day_stdev2)
    C_modes2.append(dayC_mode2)
    C_medians2.append(dayC_median2)
    C_stdevs2.append(dayC_stdev2)
    CL_medians2.append(dayCL_median2)
    CL_stdevs2.append(dayCL_stdev2)

    if Instrument == 'Vais':
        modes_wv.append(day_mode_wv)
        means_wv.append(day_mean_wv)
        medians_wv.append(day_median_wv)
        sems_wv.append(day_sem_wv)
        stdevs_wv.append(day_stdev_wv)
        C_modes_wv.append(dayC_mode_wv)
        C_medians_wv.append(dayC_median_wv)
        C_stdevs_wv.append(dayC_stdev_wv)
        CL_medians_wv.append(dayCL_median_wv)
        CL_stdevs_wv.append(dayCL_stdev_wv)


























print 'END PROGRAM'
