"""
Script to check for and output a list of missing MO files for the UKV and London model, on its-tier2

Created by Elliott Warren: Fri 09 Feb 2018
"""

import numpy as np
import os
import datetime as dt

from calendar import isleap

# -------------------------------------
# Setup
# -------------------------------------

startdir = '/data/its-tier2/micromet/data/'

savedir = '/home/micromet/Temp_Elliott/'

years = range(2016, 2018)
years_str = [str(i) for i in years]

models = {'London': 'MOLON', 'ukv': 'MOUKV'}

# ------------------------------------
# Find missing data
# ------------------------------------

# make an empty list to store the times when the files are missing
missing_dates = []
missing_dates_old_style = [] # old file format that Simone used when passing dates to Chloe
missing_files = []


for model, model_fname_part in models.iteritems():

    print 'doing model: ' + str(model)

    for year in years:

        print 'year is: ' + str(year)

        # met office directory
        modir = startdir + str(year) + '/London/L2/MetOffice/'

        # get list of DOY for the year
        if isleap(year):
            doys = range(0, 367)
        else:
            doys = range(0, 366)

        for doy in doys:

            # doy directory
            doydir = modir + 'DAY/'+ str(doy)

            # make datetime from year and doy
            date = dt.datetime.strptime(str(year)+str(doy), '%Y%j')
            date_str = date.strftime('%Y%m%d')

            # filename for KSSW from date
            filename = model_fname_part +'_FC'+date_str+'06Z_WXT_KSSW.nc'

            # check if KSSW file exists and store date + filename if not
            if os.path.isfile(doydir + filename) == False:
                missing_dates += [date_str]
                missing_files += [filename]
                missing_dates_old_style += [date.strftime('%Y-%m-%d')]

# save missing data files
np.savetxt(savedir + model + '_missing_KSSW_dates.csv', missing_dates, delimiter=',')
np.savetxt(savedir + model + '_missing_KSSW_filenames.csv', missing_files, delimiter=',')
np.savetxt(savedir + model + '_missing_KSSW_dates_oldstyle', missing_files, delimiter='')

print 'END PROGRAM'