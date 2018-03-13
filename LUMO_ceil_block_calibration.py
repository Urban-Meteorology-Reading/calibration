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

# ----------------------------
# Read in data
# ----------------------------

for day in date_range:

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

    # Read in MO data --------------

    # Get full file paths for the day and yesterday's (yest) MO data
    yest_filepath, day_filepath = lcu.mo_create_filenames(day, datadir_mo)

    # ------------------------------
    # Apply scattering correction
    # ------------------------------

    # find the cloud based on the max backscatter return, and set the backscatter at all other heights to nan
    cloud_beta = EH.find_cloud(bsc_data['backscatter'])

    # apply the multiple scattering correction for the backscatter that was not the cloud
    Scat_correct_b = lcu.scatter_correct_Vais(cloud_beta, bsc_data['range_km'])






































print 'END PROGRAM'
