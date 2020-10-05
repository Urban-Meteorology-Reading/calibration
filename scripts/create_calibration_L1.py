"""
Calibration for the LUMO ceilometers.

Code create block sets of calibration values based on a date range, which takes into account water vapour absorption.
The recommended calibration values to use are the daily modes. Saves yearly netCDF files of the outputs in L0 folder.
Based on the cumulostratus calibration method developed by Hopkin et al., in prep. Code designed mainly for CL31
and Lufft ceilometers but adapted here for use solely for LUMO ceilometers. Functions kept in the
LUMO_ceilometer_Utils.py script.

Data needed:
- L1 BSC files (UNSMOOTHED attenuated backscatter from ceilometers)
- Met Office NWP forecast files (water vapour correction: requiring, specific humidity, pressure and temperature)

 what to do to calibrate more data
L1
 1. make sure MO NWP water vapour files are present on the cluster
 2. change site_id and year below
 3. run for each site (IMU has issues on some days - known bug)
L2
 4. download CCW30 and CAL files for year AND neighbouring years if possible
 5. adjust the calibration_periods file with updates on firmware, hardware or cleaning
 6. ncview the L1 CAL files mode_wv to check if there are any sudden changes
 7. create a new 'regime' below to determine how the interpolated calibration will be estimated (block avg, trans, etc)

Created by Elliott Warren Thurs 08/03/2018
Based heavily on CloudCal_filt_VaisTemp_LUMO.py by Emma Hopkin
"""

# need to use 'module load python/canopy-2.1.3' from met-cluster. Default python env does not have the right modules... (e.g. dateutil.relativedelta)
# cd $HOME/script/cronJobs; ./qsubCron_V6.ksh -p $HOME/Temp_Elliott/scripts/calibration/TE_lumoCeilometerCalibration.ksh

# ----------------
import os
import sys
import numpy as np
import datetime as dt
import ast

#read in cli's
# proramme dir
prog_dir = sys.argv[1]
#prog_dir = 'C:\\Users\\kitbe\\Documents\\ceilometer_calibration'
# base directory for data (MM_DAILYDATA or euivalent)
base_dir = sys.argv[2]
#base_dir = 'Z:\\Tier_raw'

# years to process
yrs = sys.argv[3]
#yrs = '2018'
#sites to process
s_ids = sys.argv[4]
#s_ids = 'CL31-C_MR'

# append dir containing lcu utility library
sys.path.append(os.path.join(prog_dir, 'utils'))
import LUMO_calibration_Utils as lcu

# ----------------------------
# Setup
# ----------------------------

# ceilometers to loop through (full ceilometer ID)
site_ids = s_ids.split(';')
print(site_ids)
# years to loop through [list]
years = [int(i) for i in yrs.split(';')]
print(years)
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

        num_files_present = 0 # count how many BSC files were actually present in the year

        # create date range (daily resolution) to loop through
        # calibration values created at daily resolution
        start_date = dt.datetime(year, 1, 1)  # comparing my modes to EH modes
        end_date = dt.datetime(year, 12, 31)
        date_range = lcu.date_range(start_date, end_date, 1, 'day')

        # create simple time range (just days) for use in saving to netCDF later
        time_deltas = [i - dt.datetime(year,1,1) for i in date_range]
        date_range_netcdf = np.array([i.days for i in time_deltas])

        # ----------------------------
        # Define variables
        # ----------------------------

        # NOT water corrected calibration
        profile_total = []  #array of total number of profiles used for calibration for each day
        C_modes = []
        C_medians = []

        # water corrected calibration
        C_modes_wv = []
        C_medians_wv = []


        # loop through each day individually, create calibration coefficient and store in list variables
        for day in date_range:
            
            # 3 char DOY
            doy = day.strftime('%j')

            # state the date for each loop
            print('processing day ' + str(day))

            # ----------------------------
            # Read in data
            # ----------------------------

            # find directory name for bsc data
            datadir_bsc = os.path.join(base_dir, 'data', str(year),'London', 'L1', site, 'DAY', doy) + os.sep

            # Note: L0 BSC heights are corrected for height above ground
            #       L1 BSC heights are NOT corrected and are therefore just the range...
            bsc_filepath = os.path.join(datadir_bsc, ceil_id+'_BSC_'+site+'_'+day.strftime('%Y%j')+'_15sec.nc')
            print(bsc_filepath)
            # check if file exists
            if os.path.isfile(bsc_filepath) == True:

                try:
                    #get all the outputs
                    calib_out = lcu.create_calibration_L1(bsc_filepath, day, base_dir, cont_profs, maxB_filt, ratio_filt)
                    # add 1 to show that a file was present
                    num_files_present += 1
                    #unpack outputs
                    (dayC_mode_wv, dayC_median_wv, dayC_mode, dayC_median, no_of_profs) = tuple(calib_out)

                    #append everything
                    profile_total.append(no_of_profs)
                    C_modes.append(dayC_mode)
                    C_medians.append(dayC_median)
                    C_modes_wv.append(dayC_mode_wv)
                    C_medians_wv.append(dayC_median_wv)
                    
                except Exception as e:
                    print('create_calib_L1 failed')
                    print(e)
                    #append nans only 
                    C_modes_wv.append(np.nan)
                    C_medians_wv.append(np.nan)
                    C_modes.append(np.nan)
                    C_medians.append(np.nan)
                    profile_total.append(np.nan)

            # else if backscatter data is not available this day
            else:
                print('BSC datafile was missing')
                C_modes_wv.append(np.nan)
                C_medians_wv.append(np.nan)
                C_modes.append(np.nan)
                C_medians.append(np.nan)
                profile_total.append(np.nan)

        # if there is data to save
        if num_files_present > 0:
            # save the year's data as a netCDF file in the ANNUAL folder
            lcu.netCDF_save_calibration(C_modes_wv, C_medians_wv, C_modes, C_medians, profile_total, date_range_netcdf,
                                        site_id, site, year, base_dir + os.sep)


print('END PROGRAM')