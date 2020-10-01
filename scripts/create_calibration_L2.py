"""
Create L2 calibration product (interpolated based on L1 calibration data). Interpolatation can be against time,
window transmission or an average. Read in all the years at once for a site.

Created on 15/05/17 by EW
"""

import sys
import os
import numpy as np
import datetime as dt

#read in cli's
# proramme dir
prog_dir = sys.argv[1]
# base directory for data (MM_DAILYDATA or euivalent)
base_dir = sys.argv[2]
# years to process
yrs = sys.argv[3]
#sites to process
s_ids = sys.argv[4]

# append dir containing lcu, eu and ceil utility libraries
sys.path.append(os.path.join(prog_dir, 'utils'))

import LUMO_calibration_Utils as lcu

# ==============================================================================
# Setup
# ==============================================================================

# ceilometers to loop through (full ceilometer ID)
site_ids = s_ids.split(',')
print(site_ids)
# years to loop through [list]
years = [int(i) for i in yrs.split(',')]
print(years)

# =============================================================================
#  # EW papers    
#    
#  # paper 2 clear sky days to overplot onto the calibration
#  daystrList = ['20161125','20161129','20161130','20161204','20170120','20170122','20170325','20170408','20170526',
#     '20170828','20161102','20161205','20161227','20161229','20170105','20170117','20170118','20170119',
#     '20170121','20170330','20170429','20170522','20170524','20170601','20170614','20170615','20170619',
#     '20170620','20170626','20170713','20170717','20170813','20170827','20170902']
#  
#  # paper 3 clear sky cases
#  daystrList = ['20180216', '20180406', '20180418', '20180419', '20180420',
#  '20180505', '20180506', '20180507', '20180514', '20180515',
#  '20180519', '20180520', '20180805', '20180806', '20180902']
# 
#  new clear sky days
# clear_days = eu.dateList_to_datetime(daystrList)
# =============================================================================

# revised regime styles for paper 2 - [start, end, regime type]
regimes = {'CL31-A_KSS45W': {'1': [dt.date(2015, 9, 1), dt.date(2016, 4, 1), 'block_avg']},
           'CL31-A_IMU': {'1': [dt.date(2016, 1, 1), dt.date(2016, 8, 15), 'window_transmission'],
                          '2': [dt.date(2016, 8, 16), dt.date(2017, 5, 31), 'window_transmission'],
                          '3': [dt.date(2017, 5, 31), dt.date(2020, 9, 28), 'window_transmission']},
           'CL31-B_RGS': {'1': [dt.date(2015, 2, 5), dt.date(2016, 7, 16), 'window_transmission'],
                          '2': [dt.date(2016, 7, 17), dt.date(2017, 1, 1), 'time'],
                          '3': [dt.date(2016, 7, 17), dt.date(2017, 11, 17), 'time'],
                          '4': [dt.date(2017, 11, 17), dt.date(2020, 9, 28), 'window_transmission']},
           'CL31-C_MR': {'1': [dt.date(2015, 2, 5), dt.date(2016, 7, 28), 'block_avg'],
                         '2': [dt.date(2016, 7, 28), dt.date(2018, 12, 31), 'block_avg'],
                         '3': [dt.date(2019, 1, 1), dt.date(2020, 9, 28), 'window_transmission']},
           'CL31-D_NK': {'1': [dt.date(2011, 6, 5), dt.date(2012, 5, 1), 'block_avg'],
                         '2': [dt.date(2014, 1, 1), dt.date(2015, 5, 5), 'window_transmission'],
                         '3': [dt.date(2015, 5, 5), dt.date(2018, 12, 31), 'window_transmission']},
           'CL31-D_SWT': {'1': [dt.date(2017, 11, 10), dt.date(2017, 12, 31), 'time'],
                          '2': [dt.date(2018, 1, 1), dt.date(2020, 9, 28), 'window_transmission']},
           'CL31-E_NK': {'1': [dt.date(2016, 7, 7), dt.date(2018, 12, 31), 'window_transmission']},
           'CL31-E_HOP' : {'1': [dt.date(2018, 12, 12), dt.date(2020, 9, 28), 'window_transmission']}}
# regimes = {'CL31-A_KSS45W': {'1': [dt.date(2015, 2, 24), dt.date(2015, 6, 20), 'time'],
#                              '2': [dt.date(2015, 9, 1), dt.date(2016, 4, 1), 'block_avg']},
#            'CL31-A_IMU': {'1': [dt.date(2016, 1, 1), dt.date(2016, 8, 15), 'window_transmission'],
#                           '2': [dt.date(2016, 8, 16), dt.date(2017, 5, 31), 'window_transmission'],
#                           '3': [dt.date(2017, 5, 31), dt.date(2018, 12, 31), 'window_transmission']},
#            'CL31-B_RGS': {'1': [dt.date(2015, 2, 5), dt.date(2016, 7, 16), 'window_transmission'],
#                           '2': [dt.date(2016, 7, 17), dt.date(2017, 1, 1), 'time'],
#                           '3': [dt.date(2016, 7, 17), dt.date(2017, 11, 17), 'time'],
#                           '4': [dt.date(2017, 11, 18), dt.date(2018, 6, 1), 'block_avg'],
#                           '5': [dt.date(2018, 6, 1), dt.date(2018, 12, 31), 'window_transmission']},
#            'CL31-C_MR': {'1': [dt.date(2015, 2, 5), dt.date(2016, 7, 28), 'block_avg'],
#                          '2': [dt.date(2016, 7, 28), dt.date(2018, 12, 31), 'block_avg']},
#            'CL31-D_NK': {'1': [dt.date(2011, 6, 5), dt.date(2012, 5, 1), 'block_avg'],
#                          '2': [dt.date(2014, 1, 1), dt.date(2015, 5, 5), 'window_transmission'],
#                          '3': [dt.date(2015, 5, 5), dt.date(2018, 12, 31), 'window_transmission']},
#            'CL31-D_SWT': {'1': [dt.date(2017, 11, 10), dt.date(2017, 12, 31), 'block_avg']},
#            'CL31-E_NK': {'1': [dt.date(2016, 7, 7), dt.date(2018, 12, 31), 'window_transmission']}}

calib_pro = {}

# for each site
for site_id in site_ids:

    # get site information from site_id
    ceil_id = site_id.split('_')[0]
    ceil_type = site_id.split('-')[0]
    site = site_id.split('_')[-1]

    print '... Processing ' + site        

    #define period dir         
    perioddatadir = os.path.join(prog_dir, 'ceilometer_periods')    

    # ==============================================================================
    # Read data
    # ==============================================================================

    # read periods
    periods = lcu.read_periods(perioddatadir + os.sep, site_id)
    
    # read in all L1 calibration files
    calib_filepaths = [os.path.join(base_dir, 'data', str(i), 'London', 'L1', site, 'ANNUAL', ceil_id +'_CAL_'+site+'_KB_'+str(i)+'.nc') for i in years]
    calib = lcu.netCDF_read(calib_filepaths)
    
    # save L2 calibration data (netCDF)
    # save each year into a different file
    for year in years:  # which years to save
        print('Year: ' + str(year))
        #directory for L1 calibration file
        ceildatadir = os.path.join(base_dir, 'data', str(year), 'London', 'L1', site, 'ANNUAL')
        #directory to save L2 calibration file
        L2calsavedir = os.path.join(base_dir, 'data', str(year), 'London', 'L2', site, 'ANNUAL')
        #directory to 'DAY' folder for site and year
        ceilclouddatadir = os.path.join(base_dir, 'data', str(year), 'London', 'L1', site, 'DAY')
        
        date_range = lcu.date_range(dt.datetime(year, 1, 1), dt.datetime(year, 12, 31), 1, 'days')
        
        # list of all files for this site
        ccw30_filepaths = [os.path.join(ceilclouddatadir, i.strftime('%j'), ceil_id + '_CCW30_' + site + '_' + i.strftime('%Y%j') + '_15min.nc') for i in date_range]
        # get window transmission
        window_trans = lcu.netCDF_read(ccw30_filepaths, skip_missing_files=True)

        # ==============================================================================
        # Process
        # ==============================================================================

        # turn datetimes into dates
        calib_dates_days = np.array([i.date() for i in calib['time']])            

        # create  time series of dailymax window transmission
        window_trans_daily = lcu.calc_daily_window_trans(window_trans, calib_dates_days, calib)

        # remove event days from window transmission (firmware, hardware, cleaing etc)
        window_trans_daily = lcu.remove_events(periods, window_trans_daily)

        # process the calibration based on ceilometer 'regime' so there is data for all days.
        window_trans_daily = lcu.process_calibration_for_all_days(window_trans_daily, regimes, site_id)
        
        #save
        lcu.netCDF_save_calibration_L2(window_trans_daily, site_id, year, L2calsavedir + os.sep)

        # plot scatter of c_pro and window transmission
        # plot_cal_vs_window_trans_timeseries(window_trans_daily, savedir, site_id, clear_days, plot_clear=True)
        # plot_cal_vs_window_trans_timeseries(window_trans_daily, savedir, site_id, clear_days, plot_clear=False)
        #plot_cal_vs_window_trans_timeseries(window_trans_daily, savedir, site_id, plot_trans=False)
        #plot_cal_vs_window_trans_timeseries(window_trans_daily, savedir, site_id, plot_trans=True)


print 'END PROGRAM'
