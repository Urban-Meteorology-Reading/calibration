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

savedir = '/home/micromet/Temp_Elliott/missing_mo_files/'

years = range(2016, 2018)
years_str = [str(i) for i in years]

models = {'London': 'MOLON', 'ukv': 'MOUKV'}

# flag so that the check against all models is only done once
check_all_models = 1

# ------------------------------------
# Find missing data
# ------------------------------------

# make an empty list to store the times when all the model files are missing for a day
missing_dates_all = []
missing_dates_old_style_all = []  # old file format that Simone used when passing dates to Chloe
missing_yeardoy_all = []

extra =[]

for model, model_fname_part in models.iteritems():

    # make an empty list to store the times when the files are missing
    missing_dates = []
    missing_dates_old_style = []  # old file format that Simone used when passing dates to Chloe
    missing_files = []
    missing_yeardoy = []

    print 'doing model: ' + str(model)

    for year in years:

        print 'year is: ' + str(year)

        # met office directory
        modir = startdir + str(year) + '/London/L2/MetOffice/'

        # get list of DOY for the year
        if isleap(year):
            doys = range(1, 367)
        else:
            doys = range(1, 366)

        for doy in doys:

            doy_str = '%003d' % doy

            # doy directory
            doydir = modir + 'DAY/'+ doy_str +'/'

            # make datetime from year and doy
            date = dt.datetime.strptime(str(year)+doy_str, '%Y%j')
            date_str = date.strftime('%Y%m%d')

            # 1. check for this current model

            # filename for KSSW from date
            filename = model_fname_part +'_FC'+date_str+'06Z_WXT_KSSW.nc'

            # check if KSSW file exists and store date + filename if not
            if os.path.isfile(doydir + filename) == False:
                missing_dates += [date_str]
                missing_files += [filename]
                missing_dates_old_style += [date.strftime('%Y-%m-%d')]
                missing_yeardoy += [date.strftime('%Y-%j')]

            # 2. check across models
            if check_all_models == 1:
                # filename for KSSW from date
                fullfilepaths = [doydir + part + '_FC' + date_str + '06Z_WXT_KSSW.nc' for part in models.values()]

                # extra += [fullfilepaths]

                # booleon if they are missing or not
                full_path_missing = [os.path.isfile(f) for f in fullfilepaths]

                # if none of the model files exist for this day then store the date
                if True not in full_path_missing:
                    missing_dates_all += [date_str]
                    missing_dates_old_style_all += [date.strftime('%Y-%m-%d')]
                    missing_yeardoy_all += [date.strftime('%Y-%j')]

    # turn off flag once this model is done so it was only done once
    check_all_models = 0

    # save missing data files for each model
    np.savetxt(savedir + model + '_missing_KSSW_dates.csv', missing_dates, fmt='%s',delimiter=',')
    np.savetxt(savedir + model + '_missing_KSSW_filenames.csv', missing_files, fmt='%s',delimiter=',')
    np.savetxt(savedir + model + '_missing_KSSW_year-doy.csv', missing_yeardoy, fmt='%s', delimiter=',')
    np.savetxt(savedir + model + '_missing_KSSW_dates_oldstyle.txt', missing_dates_old_style, fmt='%s', delimiter='')

# save missing data files if True across all models
np.savetxt(savedir + 'all_missing_KSSW_dates.csv', missing_dates_all, fmt='%s',delimiter=',')
np.savetxt(savedir + 'all_missing_KSSW_year-doy.csv', missing_yeardoy_all, fmt='%s', delimiter=',')
np.savetxt(savedir + 'all_missing_KSSW_dates_oldstyle.txt', missing_dates_old_style_all, fmt='%s', delimiter='')

print 'END PROGRAM'