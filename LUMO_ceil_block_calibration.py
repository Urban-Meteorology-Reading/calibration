"""
Calibration for the LUMO ceilometers.

Code create block sets of calibration values based on a date range, which takes into account water vapour absorption.
The recommended calibration values to use are the daily modes. Saves yearly netCDF files of the outputs in L0 folder.
Based on the cumulostratus calibration method developed by Hopkin et al., in prep. Code designed mainly for CL31
and Lufft ceilometers but adapted here for use solely for LUMO ceilometers. Functions kept in the
LUMO_ceilometer_Utils.py script.

Data needed is:
- L1 BSC files (UNSMOOTHED attenuated backscatter from ceilometers)
- L0 CCW30 files (transmission data)
- Met Office NWP forecast files (water vapour correction: requiring, specific humidity, pressure and temperature)

Created by Elliott Warren Thurs 08/03/2018
Based heavily on CloudCal_filt_VaisTemp_LUMO.py by Emma Hopkin

ToDo list:
- code works when output is compared to that produced by EH from their original script
- code wants a 'tidy up' as several variables and steps are not needed for the LUMO processing
"""

import sys
# append dir containing utility library
sys.path.append('C:/Users/Elliott/Documents/PhD Reading/LUMO - Sensor network/calibration/utils')
import LUMO_calibration_Utils as lcu

import os # operating system library to issue Unix commands
import numpy as np # numerical python library for arrays and mathematical functions
import datetime as dt
import ellUtils as eu
import ceilUtils as ceil
from netCDF4 import Dataset


def netCDF_save_calibration(C_modes_wv, C_medians_wv, C_modes, C_medians, profile_total, date_range_netcdf,
                            site_id, site, year):
    """
    Save the year's calibration data in a netCDF file. Store it in the ANNUAL folder
    :param C_modes_wv: wv is with water vapour correction
    :param C_medians_wv:
    :param C_modes:
    :param C_medians:
    :param profile_total:
    :param date_range_netCDF: days since 1st Jan of the year being processed
    :param site_id: full site id e.g. 'CL31-A_KSS45W'
    :param site: site name e.g. 'KSS45W'
    :param year: year processed [int]
    :return:

    EW 13//04/18
    """

    # Create save file id (put CAL in the id)
    a = site_id.split('_')
    site_save_id = a[0] + '_CAL_' + a[1]

    ncsavedir = '/data/its-tier2/micromet/data/'+year+'/London/L1/'+site+'/ANNUAL/'

    # Create save filename
    ncfilename = site_save_id + '_' + str(year) + '.nc'

    # Create netCDF file
    ncfile = Dataset(ncsavedir + '/' + ncfilename, 'w')

    # Create dimensions
    ncfile.createDimension('time', len(date_range))

    # Create co-ordinate variables
    nc_time = ncfile.createVariable('time', np.float64, ('time',))
    nc_time[:] = date_range_netcdf  # days since 1st Jan of this year
    nc_time.units = 'days since ' + dt.datetime(year, 1, 01).strftime('%Y-%m-%d %H:%M:%S')

    # Create main variables
    nc_cal_mode_wv = ncfile.createVariable('CAL_mode_wv', np.float64, ('time',))
    nc_cal_mode_wv[:] = C_modes_wv
    nc_cal_mode_wv.long_name = 'modal calibration coefficient with water vapour correction'

    nc_cal_median_wv = ncfile.createVariable('CAL_median_wv', np.float64, ('time',))
    nc_cal_median_wv[:] = C_medians_wv
    nc_cal_median_wv.long_name = 'median calibration coefficient with water vapour correction'

    nc_cal_mode = ncfile.createVariable('CAL_mode', np.float64, ('time',))
    nc_cal_mode[:] = C_modes
    nc_cal_mode.long_name = 'modal calibration coefficient without water vapour correction'

    nc_cal_median = ncfile.createVariable('CAL_median', np.float64, ('time',))
    nc_cal_median[:] = C_medians
    nc_cal_median.long_name = 'median calibration coefficient without water vapour correction'

    nc_profile_total = ncfile.createVariable('profile_total', np.float64, ('time',))
    nc_profile_total[:] = profile_total

    # Extra attributes
    ncfile.history = 'Created ' + dt.datetime.now().strftime('%Y-%m-%d %H:%M') + ' GMT'
    ncfile.site_id = site_id

    # close file
    ncfile.close()

    # print status
    print ncfilename + ' save successfully!'
    print ''

    return

# compare agaisnt values produced with this code
def load_old_kss45w_c_wv():
    """
    Quick load the old ceilometer calibration values done by EH the first time around, to compare with those being made
    by this script.
    :return:
    """

    import pickle

    calib_path = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/' \
                 'Calibrations_for_LUMO_Ceilometers/'

    filename = calib_path + 'CL31-A_KSS45W' + '_window_trans_daily_cpro.pickle'

    # sort site name out (is in CL31-A_BSC_KSS45W format, but needs to be CL31-A_KSS45W

    # load calibration data (using pickle)
    with open(filename, 'rb') as handle:
        c_wv_old_kss45w = pickle.load(handle)

    return c_wv_old_kss45w
def load_old_nk_c_wv():
    """
    Quick load the old ceilometer calibration values done by EH the first time around, to compare with those being made
    by this script.
    :return:
    """

    import pickle

    calib_path = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/' \
                 'Calibrations_for_LUMO_Ceilometers/'

    filename = calib_path + 'CL31-D_NK' + '_window_trans_daily_cpro.pickle'

    # sort site name out (is in CL31-A_BSC_KSS45W format, but needs to be CL31-A_KSS45W

    # load calibration data (using pickle)
    with open(filename, 'rb') as handle:
        c_wv_old_kss45w = pickle.load(handle)

    return c_wv_old_kss45w

# ----------------------------
# Setup
# ----------------------------

# ceilometers to loop through (full ceilometer ID)
site_ids = ['CL31-A_KSS45W', 'CL31-A_IMU', 'CL31-B_RGS', 'CL31-C_MR', 'CL31-D_NK', 'CL31-D_SWT', 'CL31-E_NK']


# years to loop through [list]
years = [2015]

# directories
# final datadir structure needs to be...
# /data/its-tier2/micromet/data/2017/London/L1/IMU/DAY/200/
# NOTE: Make sure bsc files are from L0!! - EH cubing does it on unsmoothed L1 files, so do it on those.

# datadir_bsc = 'C:/Users/Elliott/Documents/PhD Reading/LUMO - Sensor network/calibration/data/2015/London/L1/01/'
#datadir_ccw30 = 'C:/Users/Elliott/Documents/PhD Reading/LUMO - Sensor network/calibration/data/2015/01/'
#datadir_mo ='C:/Users/Elliott/Documents/PhD Reading/LUMO - Sensor network/calibration/data/MO/'

# netCDF savedir
ncsavedir = 'C:/Users/Elliott/Documents/PhD Reading/LUMO - Sensor network/calibration/data/ncsave/'

# time to loop through
# start_date = dt.datetime(2014, 11, 27)
# end_date = dt.datetime(2014, 11, 28)

# # test day against EH cubed day '04.nc'
# start_date = dt.datetime(2015, 1, 04)
# end_date = dt.datetime(2015, 1, 05)

# settings to tune calibration
ratio_filt = 0.05
maxB_filt = -10  #set high so that this filter is no longer used
cont_profs = 5  #number of continuous profiles required for calibration (must be odd no.)

# loop through site ids
for site_id in site_ids:

    # get site information from site_id
    ceil_id = site_id.split('_')[0]
    ceil_type = site_id.split('-')[0]
    site = site_id.split('_')[-1]

    # loop through years - create 1 netCDF file per year, per site
    for year in years:

        # create date range (daily resolution) to loop through
        # calibration values created at daily resolution
        start_date = dt.datetime(year, 1, 01)  # comparing my modes to EH modes
        end_date = dt.datetime(year, 12, 31)
        # start_date = dt.datetime(year, 1, 02)  # comparing my modes to EH modes
        # end_date = dt.datetime(year, 1, 12)
        date_range = eu.date_range(start_date, end_date, 1, 'day')

        # create simple time range (just days) for use in saving to netCDF later
        time_deltas = [i - dt.datetime(year,1,01) for i in date_range]
        date_range_netcdf = np.array([i.days for i in time_deltas])

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

        index_of_maxB = []  #index of max backscatter for each profile (includes all meteorological conditions)
        value_of_maxB = []  #value of max backscatter for each profile

        Profile_integrations = []       #integral of each profile (0.1-2.4km)

        window_tau = []     #instrument reported window transmission for each profile [%]
        window_tau_alert=[] #marker if window transmission falls below 90%
        pulse_energy = []   #intrument reported pulse energy for each profile [%]
        pulse_energy_alert = []#marker if pulse energy falls below 90%
        CBH = []            #instrument reported cloud base height [km] - no data => -999
        All_S = []          #apparent S for each profile
        S_box = []          #apparent S in arrays for box plot of each day

        WV_trans = []       #transmission correction by profile from MWR
        lengths = []
        daily_WV = []       #transmission correction by day from MWR
        daymean_modelWV = []#transmission correction by day from model
        All_modelWV = []    #transmission correction by profile from model

        profiles_in_row = []
        file_locs = []

        # # load KSS45W and NK calibrations by Emma to compare with
        # c_wv_old_kss45w = load_old_kss45w_c_wv()
        # c_wv_old_nk = load_old_nk_c_wv()

        # loop through each day individually, create calibration coefficient and store in list variables

        for day in date_range:

            # 3 char DOY
            doy = day.strftime('%j')

            # state the date for each loop
            print 'processing day ' + str(day)

            # ----------------------------
            # Read in data
            # ----------------------------

            # find directory name for bsc data
            datadir_bsc = '/data/its-tier2/micromet/data/'+str(year)+'/London/L1/'+site+'/DAY/' + doy + '/'

            # Note: L0 BSC heights are corrected for height above ground
            #       L1 BSC heights are NOT corrected and are therefore just the range...
            bsc_filepath = datadir_bsc + ceil_id+'_BSC_'+site+'_'+day.strftime('%Y%j')+'_15sec.nc'
            # bsc_filepath = datadir_bsc + 'CL31-D_BSC_NK_'+day.strftime('%Y%j')+'_15sec.nc'

            # check if file exists
            if os.path.isfile(bsc_filepath) == True:

                # read in L1 unscmoothed backscatter data (do not correct for SNR)
                bsc_data, _ = ceil.netCDF_read_BSC(bsc_filepath, var_type='beta', SNRcorrect=False)

                # transpose the backscatter for EH functions
                bsc_data['backscatter'] = np.transpose(bsc_data['backscatter'])

                # create range in [km]
                bsc_data['range_km'] = bsc_data['range'] / 1000.0

                # ------------------------------
                # Apply scattering correction
                # ------------------------------

                # find the cloud based on the max backscatter return, and set the backscatter at all other heights to nan
                cloud_beta = lcu.find_cloud(bsc_data['backscatter'])

                # apply the multiple scattering correction for the backscatter that was not the cloud
                Scat_correct_b = lcu.scatter_correct_Vais(cloud_beta, bsc_data['range_km'])

                # apply the multiple scattering to correct the non-cloud backscatter,
                #    and merge the array with the cloud backscatter array
                beta_arr = lcu.corr_beta(Scat_correct_b, bsc_data['backscatter'])

                # ----------------------------------------------
                # Apply water vapour attenuation correction
                # ----------------------------------------------

                # get yesterday's time to get the right forecast file for the water vapour
                yest = day - dt.timedelta(days=1)

                # Get full file paths for the day and yesterday's (yest) MO data
                yest_filepath = lcu.mo_create_filename(yest)
                day_filepath = lcu.mo_create_filename(day)

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

                ## 1. Calculate lidar ratio (S) without water vapour correction

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

                ## 2. Calculate S with water vapour correction

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
                Cal_hist, no_of_profs = lcu.get_counts(Step2_S)  # Histogram of filtered S
                no_in_peak, day_mode, day_mean, day_median, day_sem, day_stdev, dayC_mode, dayC_median, dayC_stdev, dayCL_median, dayCL_stdev = lcu.S_mode_mean(
                Step2_S, Cal_hist)

                Cal_hist_wv, no_of_profs_wv = lcu.get_counts(Step2_S_wv)  # Histogram of filtered S
                no_in_peak_wv, day_mode_wv, day_mean_wv, day_median_wv, day_sem_wv, day_stdev_wv, dayC_mode_wv, dayC_median_wv, dayC_stdev_wv, dayCL_median_wv, dayCL_stdev_wv = lcu.S_mode_mean(
                Step2_S_wv, Cal_hist_wv)

                Cal_hist2, no_of_profs2 = lcu.get_counts(Step2_S2)  # Histogram of filtered S
                no_in_peak2, day_mode2, day_mean2, day_median2, day_sem2, day_stdev2, dayC_mode2, dayC_median2, dayC_stdev2, dayCL_median2, dayCL_stdev2 = lcu.S_mode_mean(
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

        # save the year's data as a netCDF file in the ANNUAL folder
        netCDF_save_calibration(C_modes_wv, C_medians_wv, C_modes, C_medians, profile_total, date_range_netcdf,
                                    site_id, site, year)



# # quick compare of the old calibrated data and the ones made by this script
# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot_date(c_wv_old_kss45w['dates'][:20], c_wv_old_kss45w['c_wv'][:20], label='old modes (EH)')
# plt.plot_date(date_range, C_modes_wv, label='new modes (EW)')
# plt.plot_date(date_range, C_medians, label='new median no wv')
# plt.plot_date(date_range, C_medians_wv, label='new med')
# plt.plot_date(date_range, C_stdevs_wv, label='new stdev')
# plt.legend()


# date_range[4]
# C_modes_wv[4]
#vs
# A_KSS45W.Dates[1410]
# A_KSS45W.C_modes_wv[1410]























print 'END PROGRAM'
