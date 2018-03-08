"""
Calibration for the LUMO ceilometers (designed mainly for CL31 and Lufft ceilometers).

Code create block sets of calibration values, which takes into account water vapour absorption. Saves yearly
netCDF files of the outputs in L0 folder. Based on the cumulostratus calibration method designed by
Hopkin et al., in prep.

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
# import LoadData_Utils as LD
import EH_Utils as EH
import Cal_Utils as CAL
import Stats_Utils as ST

import os # operating system library to issue Unix commands
from netCDF4 import Dataset # to read standard netCDF files
import numpy as np # numerical python library for arrays and mathematical functions
import datetime as dt
import copy
import ellUtils as eu

# ----------------------------
# Setup
# ----------------------------

# directories
# final datadir structure needs to be...
# /data/its-tier2/micromet/data/2017/London/L1/IMU/DAY/200/
datadir_bsc = 'C:/Users/Elliott/Documents/PhD Reading/LUMO - Sensor network/calibration/data/2015/01/'
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

    yesterday = day - dt.timedelta(days=1)

    # get MO filepaths for the water vapour
    yesterday_filename = 'MOUKV_FC'+yesterday.strftime('%Y%m%d')+'06Z_WXT_KSSW.nc'
    day_filename = 'MOUKV_FC'+day.strftime('%Y%m%d')+'06Z_WXT_KSSW.nc'

    yesterday_filepath = datadir_mo + yesterday_filename
    day_filepath = datadir_mo + day_filename

