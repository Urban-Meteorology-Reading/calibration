"""
Create L2 calibration product (interpolated based on L1 calibration data). Interpolatation can be against time,
window transmission or an average. Read in all the years at once for a site.

Created on 15/05/17 by EW
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.dates import DateFormatter
# from pm_RH_vs_mod_obs_backscatter import dateList_to_datetime

import numpy as np
import datetime as dt
import pickle

import ellUtils as eu
import ceilUtils as ceil

# Read

def read_cal():

    """Read calibration data"""

    sites = {'CL31-A_KSS45W':cal.A_KSS45W,'CL31-B_RGS':cal.B_RGS,'CL31-C_MR':cal.C_MR, 'CL31-D_NK':cal.D_NK}

    calib = {}
    for key, item in sites.iteritems():
        # read in calibration

        calib[key] = {'Dates': item.Dates, 'wv_cal': np.array(item.C_modes_wv),
                      'samples': np.array(item.profile_total)}

    return calib

def read_periods(perioddatadir, site):

    """ Read in ceilometer period data, e.g. firmware, hardware, window transmission and pulse changes"""

    datapath = perioddatadir + 'ceilometer_periods_' + site + '.csv'
    # read in ceilometer periods
    # periods = np.genfromtxt(datapath, delimiter=',')
    periods_raw = eu.csv_read(datapath)

    # sort data out into dictionary
    periods = {}
    for i in np.arange(len(periods_raw[0])):
        periods[periods_raw[0][i]] = [j[i] for j in periods_raw[1:]]

    # convert date strings into datetimes
    periods['Date'] = [dt.datetime.strptime(d, '%d/%m/%Y') for d in periods['Date']]

    return periods

def read_window_trans(site, datadir):

    # site id (short)
    site_id = site.split('_')[-1]
    ceil_id = site.split('_')[0]

    # get filename
    fname1 = datadir + ceil_id + '_CCW30_' + site_id + '_2015_15min.nc'
    fname2 = datadir + ceil_id + '_CCW30_' + site_id + '_2016_15min.nc'

    # read window transmission
    transmission1 = ceil.netCDF_read_CCW30(fname1, vars='transmission')
    transmission2 = ceil.netCDF_read_CCW30(fname2, vars='transmission')

    # merge dictionaries
    # transmission = eu.merge_dicts(transmission1, transmission1)

    transmission = {}
    for var, var_data in transmission1.iteritems():

        if var == 'time':
            transmission[var] = var_data + transmission2[var]
        elif (var == 'level_height') | (var == 'height'):
            transmission[var] = var_data
        else:
            # transmission[var] = np.ma.hstack((transmission1[var], np.ma.masked_array(transmission2[var],mask=np.zeros(len(transmission2[var])))))
            transmission[var] = np.hstack((transmission1[var], transmission2[var]))



    return transmission

def read_pulse(site, datadir):

    # site id (short)
    site_id = site.split('_')[-1]
    ceil_id = site.split('_')[0]

    # get filename
    fname1 = datadir + ceil_id + '_CCW30_' + site_id + '_2015_15min.nc'
    fname2 = datadir + ceil_id + '_CCW30_' + site_id + '_2016_15min.nc'

    # read window transmission
    transmission1 = ceil.netCDF_read_CCW30(fname1, vars='pulse')
    transmission2 = ceil.netCDF_read_CCW30(fname2, vars='pulse')

    # merge dictionaries
    # transmission = eu.merge_dicts(transmission1, transmission1)

    transmission = {}
    for var, var_data in transmission1.iteritems():

        if var == 'time':
            transmission[var] = var_data + transmission2[var]
        elif (var == 'level_height') | (var == 'height'):
            transmission[var] = var_data
        else:
            # transmission[var] = np.ma.hstack((transmission1[var], np.ma.masked_array(transmission2[var],mask=np.zeros(len(transmission2[var])))))
            transmission[var] = np.hstack((transmission1[var], transmission2[var]))


    return transmission

# process

def remove_events(periods, window_trans_daily):

    # remove days in window_trans_daily when events happened (firmware change, window cleaning etc)
    period_event_days = np.array([i.date() for i in periods['Date']])

    for day in period_event_days:
        idx = np.where(np.array(window_trans_daily['dates']) == day)

        # turn daily data into nan as it is unreliable
        for key in window_trans_daily.iterkeys():
            if key != 'dates':
                window_trans_daily[key][idx] = np.nan

    return window_trans_daily

def calc_daily_window_trans(window_trans, calib_dates_days, calib):

    """
    Calculate the daily maximum window transmission, and pair it with the calibration coefficient that factors in
    water vapour.
    :param window_trans:
    :param calib_dates_days:
    :param calib_raw:
    :return:
    """

    # find all unique days in window_trans time
    # loop through days
    # find all dates with that day
    # max(c) and store

    dayMin = window_trans['time'][0].date()
    dayMax = window_trans['time'][-1].date()
    # np.array([i.date() for i in window_trans['time']])

    window_trans_dates = np.array([i.date() for i in window_trans['time']])

    daysRange = eu.date_range(dayMin, dayMax, 1, 'days')
    window_trans_daily = {'dates': daysRange, 'max_window': np.empty(len(daysRange)),
                          'c_wv': np.empty(len(daysRange)), 'samples': np.empty(len(daysRange))}
    window_trans_daily['c_wv'][:] = np.nan
    window_trans_daily['max_window'][:] = np.nan
    window_trans_daily['samples'][:] = np.nan

    for day, dayIdx in zip(daysRange, np.arange(len(daysRange))):

        # idx for all days on this day
        winIdx = np.where(window_trans_dates == day)

        # store maximum window transmission for the day
        if winIdx[0].size != 0:
            window_trans_daily['max_window'][dayIdx] = np.nanmax(window_trans['transmission'][winIdx])

        # get c
        cIdx = np.where(calib_dates_days == day)

        # store c for the day
        if cIdx[0].size != 0:
            window_trans_daily['c_wv'][dayIdx] = calib['wv_cal'][cIdx]


        # get c
        sIdx = np.where(calib_dates_days == day)

        # store c for the day
        if sIdx[0].size != 0:
            window_trans_daily['samples'][dayIdx] = calib['samples'][sIdx]


    return window_trans_daily

def process_calibration_for_all_days(window_trans_daily, regimes, site):

    """
    Create the processed calibration data, which fills in missing days. Filling type depends
    on the 'regime', whether it should be a simple block average, fit over time or a regression fit
    against window transmission

    :param window_trans_daily:
    :param regimes:
    :param site:
    :return:
    """

    window_trans_daily['c_pro'] = np.empty(len(window_trans_daily['dates']))
    window_trans_daily['c_pro'][:] = np.nan

    # fill calib_pro ready for processed values
    # days = eu.date_range(dt.datetime(2015, 2, 5), dt.datetime(2016, 12, 31), 1, 'days')
    # calib_pro[site] = {'dates': np.array([i.date() for i in days]),
    #                    'c_pro': np.empty(len(days))}
    # calib_pro[site]['c_pro'][:] = np.nan

    # create the processed calibration coeffs
    for reg, values in regimes[site].iteritems():

        # find matching dates to process over
        whole_idx = np.where((np.array(window_trans_daily['dates']) > values[0]) &
                             (np.array(window_trans_daily['dates']) < values[1]))

        # KSS45W
        if values[2] == 'time':

            # can do this with [x] as days are equally spaced
            x = whole_idx[0]
            y = window_trans_daily['c_wv'][whole_idx]
            days = np.array(window_trans_daily['dates'])[whole_idx]

            idx_i = np.where(~np.isnan(x) & ~np.isnan(y))

            m, b = np.polyfit(x[idx_i], y[idx_i], 1)

            for x_i in whole_idx:
                window_trans_daily['c_pro'][x_i] = (m * x_i) + b

        # RGS and NK_E
        if values[2] == 'window_transmission':

            # get data for this period
            x = window_trans_daily['max_window'][whole_idx]
            y = window_trans_daily['c_wv'][whole_idx]

            # find data where neither value in the pair is nan
            idx_i = np.where(~np.isnan(x) & ~np.isnan(y))

            # find eq
            m, b = np.polyfit(x[idx_i], y[idx_i], 1)

            for x_i in whole_idx:
                window_trans_daily['c_pro'][x_i] = (m * window_trans_daily['max_window'][x_i]) + b

        if values[2] == 'block_avg':
            window_trans_daily['c_pro'][whole_idx] = np.nanmean(window_trans_daily['c_wv'][whole_idx])


    return window_trans_daily

# plotting

def plot_hist(window_trans_daily):

    """very fast histogram plot of sample size"""

    data = window_trans_daily['samples'][~np.isnan(window_trans_daily['samples'])]
    fig = plt.figure()
    plt.hist(data, bins=50)
    plt.savefig(savedir + '/' + site + '/samplesize.png')
    plt.close(fig)

    return

def plot_smooth(dates, calibration, ma_7, ma_10, ma_30):

    """
    Plot the smoothed curves...
    :param dates:
    :param calibration:
    :param ma_7:
    :param ma_10:
    :param ma_30:
    :return:
    """

    # plot
    fig = plt.figure()
    plt.plot_date(dates, calibration, label='linear interp.', fmt='-', ls='-')
    # plt.plot_date(dates, raw, label='raw', fmt='-', ls='-')
    plt.plot_date(dates, ma_7, label='7 day', fmt='-', ls='-')
    plt.plot_date(dates, ma_10, label='10 day', fmt='-', ls='-')
    plt.plot_date(dates, ma_30, label='30 day', fmt='-', ls='-')

    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%m/%y'))
    ax.set_xlim([dt.datetime(2015, 2, 1), dt.datetime(2016, 6, 1)])
    ax.set_xlabel('Time [mm/YY]')
    ax.set_ylabel('water vapour C_modes')
    plt.suptitle(site)
    plt.legend()
    plt.savefig(savedir + 'calibration_wv_' + site + '.png')
    plt.close(fig)

    return




def plot_cal_wind_pulse_timeseries_new_cal(window_trans_daily, savedir, site, clear_days):

    """ Plot time series of points, of raw calibration, window transmission and pulse energy"""

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()
    marker = 4
    ax1.plot_date(window_trans_daily['dates'], window_trans_daily['c_wv'], label='raw', color='g', markersize=marker)
    ax1.plot_date(window_trans_daily['dates'], window_trans_daily['c_pro'], label='c_pro', color='r', markersize=marker)
    ax2.plot_date(window_trans['time'], window_trans['transmission'], label='window_trans', markersize=marker)
    # ax2.plot_date(pulse['time'], pulse['pulse'], label='pulse', markersize=marker)

    # put vertical lines in for each of the clear days
    for day in clear_days:
        ax2.plot_date([day, day], [-100, 200], fmt='-', ls='--', color='black')

    # Prettify
    ax2.set_ylim([0.0, 100.0])
    ax = plt.gca()
    ax.xaxis.set_major_formatter(DateFormatter('%m/%y'))
    ax.set_xlim([window_trans['time'][0], window_trans['time'][-1]])
    ax.set_xlabel('Time [mm/YY]', labelpad=0)
    ax1.set_ylabel('water vapour C')
    ax2.set_ylabel('Percentage')
    plt.suptitle(site)
    plt.legend()
    plt.savefig(savedir + 'rawCalib_trans_proCalib' + site + '.png')
    plt.close(fig)

def plot_cal_wind_pulse_timeseries(calib_dates, calib_raw, window_trans, pulse, savedir, site, clear_days):

        """ Plot time series of points, of raw calibration, window transmission and pulse energy"""

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax2 = ax1.twinx()
        marker = 4
        ax1.plot_date(calib_dates, calib_raw, label='raw', color='g', markersize=marker)
        ax2.plot_date(window_trans['time'], window_trans['transmission'], label='window_trans', markersize=marker)
        ax2.plot_date(pulse['time'], pulse['pulse'], label='pulse', markersize=marker)

        # put vertical lines in for each of the clear days
        for day in clear_days:

            ax2.plot_date([day, day], [-100, 200], fmt='-', ls='--', color='black')

        # Prettify
        ax2.set_ylim([0.0, 100.0])
        ax = plt.gca()
        ax.xaxis.set_major_formatter(DateFormatter('%m/%y'))
        ax.set_xlim([window_trans['time'][0], window_trans['time'][-1]])
        ax.set_xlabel('Time [mm/YY]',labelpad=0)
        ax1.set_ylabel('water vapour C')
        ax2.set_ylabel('Percentage')
        plt.suptitle(site)
        plt.legend()
        plt.savefig(savedir + 'rawCalib_trans_pulse_' + site + '.png')
        plt.close(fig)

def plot_period(periods, window_trans_daily, site, savedir):

    """Plot for a period. Can swap x, y and c for window trans, sample size and calibration"""

    idx = np.where((np.array(periods['Type']) == 'Firmware') |
                   (np.array(periods['Type']) == 'Deployed') |
                   (np.array(periods['Type']) == 'Removed') |
                   (np.array(periods['Type']) == 'Present'))[0]

    if idx[0].size != 0:
        for key in periods.iterkeys():
            periods[key] = [periods[key][i] for i in idx]

    # find all window, c and sample in current date range
    periods_days = np.array([i.date() for i in periods['Date']])

    # loop through start and end day pairs
    for startDay, endDay in zip(periods_days[:-1], periods_days[1:]):

        # get idx position for this period
        # for some reason the idx array is within another array... is this to do with having two arguments?
        idx = np.where((np.array(window_trans_daily['dates']) > startDay) &
                       (np.array(window_trans_daily['dates']) < endDay))

        # if window_trans_daily data exists for this date, extract and plot the data
        if idx[0].size != 0:

            # extract data for this period
            window_trans_daily_period = {}
            for key in window_trans_daily.iterkeys():
                window_trans_daily_period[key] = [window_trans_daily[key][i] for i in idx[0]]

            # if there is data for window transmission and c then plot
            if np.any(np.isfinite(window_trans_daily_period['c_wv'])):
                # use the scatter function to make the graphs
                scatter_window_c_period(window_trans_daily_period, site, savedir, startDay, endDay)

                scatter_c_sample_period(window_trans_daily_period, site, savedir, startDay, endDay)

    return

def scatter_window_c_all(window_trans_daily, site, savedir):

    # scatter plot for the calibration data. daily max window trans vs calibration factor.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(window_trans_daily['max_window'], window_trans_daily['c_wv'], color='g')

    # add a linear fit to the plot
    m, b = eu.linear_fit_plot(window_trans_daily['max_window'], window_trans_daily['c_wv'], ax, ls='-', color='black')

    # string for equation
    eqstr = 'y = %.2fx + %.2f' % (m, b)

    # Prettify
    ax.set_ylim([np.nanmin(window_trans_daily['c_wv']), np.nanmax(window_trans_daily['c_wv'])])
    ax.set_xlim([np.nanmin(window_trans_daily['max_window']), np.nanmax(window_trans_daily['max_window'])])
    # ax.xaxis.set_major_formatter(DateFormatter('%m/%y'))
    # ax.set_xlim([dt.datetime(2015, 2, 1), dt.datetime(2016, 6, 1)])
    ax.set_xlabel('Window transmission (daily max) [%]')
    ax.set_ylabel('water vapour C')
    plt.suptitle(site + '; ' + eqstr)
    plt.legend()
    plt.savefig(savedir + 'rawCalib_vs_maxWindow_daily_' + site + '.png')
    plt.close(fig)

    return

def scatter_window_c_all_type(window_trans_daily, periods, site, savedir):

    """Scatter all the data but colour based on type"""

    def scatter_part(ax, periods, window_trans_daily):

        # find all window, c and sample in current date range
        periods_days = np.array([i.date() for i in periods['Date']])

        # loop through start and end day pairs
        for startDay, endDay in zip(periods_days[:-1], periods_days[1:]):

            # get idx position for this period
            # for some reason the idx array is within another array... is this to do with having two arguments?
            idx = np.where((np.array(window_trans_daily['dates']) > startDay) &
                                     (np.array(window_trans_daily['dates']) < endDay))

            # if window_trans_daily data exists for this date, extract and plot the data
            if idx[0].size != 0:

                # extract data for this period
                window_trans_daily_period = {}
                for key in window_trans_daily.iterkeys():
                    window_trans_daily_period[key] = [window_trans_daily[key][i] for i in idx[0]]

                # if there is data for window transmission and c then plot
                if np.any(np.isfinite(window_trans_daily_period['c_wv'])):

                    # add a linear fit to the plot
                    m, b = eu.linear_fit_plot(window_trans_daily_period['max_window'], window_trans_daily_period['c_wv'], ax, ls='-',
                                              color='black')

                    # string for equation
                    eqstr = 'y = %.2fx + %.2f' % (m, b)

                    # scatter_window_c_period(window_trans_daily_period, site, savedir, startDay, endDay)
                    ax.scatter(window_trans_daily_period['max_window'], window_trans_daily_period['c_wv'],
                                    label=startDay.strftime('%Y%m%d') + '-' + endDay.strftime('%Y%m%d') +
                                          '; ' + eqstr)



        return

    # scatter plot for the calibration data. daily max window trans vs calibration factor.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # find unique types
    # for type, find periods
    # put linear fit to data it and keep regression line
    # scatter data for that period + label date range for period + regress eq.

    scatter_part(ax, periods, window_trans_daily)

    # add a linear fit to the plot
    # m, b = eu.linear_fit_plot(window_trans_daily['max_window'], window_trans_daily['c_wv'], ax, ls='-', color='black')

    # string for equation
    #eqstr = 'y = %.2fx + %.2f' % (m, b)

    # Prettify
    ax.set_ylim([np.nanmin(window_trans_daily['c_wv']), np.nanmax(window_trans_daily['c_wv'])])
    ax.set_xlim([np.nanmin(window_trans_daily['max_window']), np.nanmax(window_trans_daily['max_window'])])
    # ax.xaxis.set_major_formatter(DateFormatter('%m/%y'))
    # ax.set_xlim([dt.datetime(2015, 2, 1), dt.datetime(2016, 6, 1)])
    ax.set_xlabel('Window transmission (daily max) [%]')
    ax.set_ylabel('water vapour C')
    plt.suptitle(site)
    plt.legend()
    plt.savefig(savedir + 'rawCalib_vs_maxWindow_daily_' + site + '.png')
    plt.close(fig)

    return

def scatter_window_c_period(window_trans_daily, site, savedir, startDay, endDay):

    # scatter plot for the calibration data. daily max window trans vs calibration factor.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    p = ax.scatter(window_trans_daily['max_window'], window_trans_daily['c_wv'], c = window_trans_daily['samples'],
                   vmin= 0, vmax=2500, cmap=cm.jet)

    # add a linear fit to the plot
    m, b = eu.linear_fit_plot(window_trans_daily['max_window'], window_trans_daily['c_wv'], ax, ls='-', color='black')

    # string for equation
    eqstr = 'y = %.2fx + %.2f' % (m, b)

    # period range as a string for saving
    period_range_str = startDay.strftime('%Y%m%d') + '-' + endDay.strftime('%Y%m%d')

    # Prettify
    ylim = [np.nanmin(window_trans_daily['c_wv']), np.nanmax(window_trans_daily['c_wv'])]
    xlim = [np.nanmin(window_trans_daily['max_window']), np.nanmax(window_trans_daily['max_window'])]

    # some periods only have 100 % window transmission, leading to weird limits being set on the xlim.
    # hence this deals with that edge case
    if xlim[0] == xlim[1]:
        xlim[0] = 90.0
        xlim[1] = 100.0

    # remove top and right axis but keep lables
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    p.set_clip_on(False)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    # ax.xaxis.set_major_formatter(DateFormatter('%m/%y'))
    # ax.set_xlim([dt.datetime(2015, 2, 1), dt.datetime(2016, 6, 1)])
    ax.set_xlabel('Window transmission (daily max) [%]')
    ax.set_ylabel('water vapour C')
    plt.suptitle(site + '; ' + eqstr + '\n' + period_range_str)
    plt.colorbar(p)
    plt.legend()

    plt.savefig(savedir + '/' + site + '/rawCalib_vs_maxWindow_daily_' + period_range_str + '_' + site + '.png')

    return

def scatter_c_sample_period(window_trans_daily, site, savedir, startDay, endDay):

    """Scatter sample vs c"""

    # scatter plot for the calibration data. daily max window trans vs calibration factor.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    p = ax.scatter(window_trans_daily['samples'], window_trans_daily['c_wv'], c = window_trans_daily['max_window'], cmap=cm.jet)

    # add a linear fit to the plot
    m, b = eu.linear_fit_plot(window_trans_daily['samples'], window_trans_daily['c_wv'], ax, ls='-', color='black')

    # string for equation
    eqstr = 'y = %.2fx + %.2f' % (m, b)

    # period range as a string for saving
    period_range_str = startDay.strftime('%Y%m%d') + '-' + endDay.strftime('%Y%m%d')

    # Prettify
    ylim = [np.nanmin(window_trans_daily['c_wv']), np.nanmax(window_trans_daily['c_wv'])]
    xlim = [np.nanmin(window_trans_daily['samples']), np.nanmax(window_trans_daily['samples'])]

    # some periods only have 100 % window transmission, leading to weird limits being set on the xlim.
    # hence this deals with that edge case
    #if xlim[0] == xlim[1]:
    #    xlim[0] = 90.0
    #    xlim[1] = 100.0

    # remove top and right axis but keep lables
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    p.set_clip_on(False)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    # ax.xaxis.set_major_formatter(DateFormatter('%m/%y'))
    # ax.set_xlim([dt.datetime(2015, 2, 1), dt.datetime(2016, 6, 1)])
    ax.set_xlabel('sample size')
    ax.set_ylabel('water vapour C')
    plt.suptitle(site + '; ' + eqstr + '\n' + period_range_str)
    plt.colorbar(p)
    plt.legend()

    plt.savefig(savedir + '/' + site + '/rawCalib_vs_sample_daily_' + period_range_str + '_' + site + '.png')

    return

def dateList_to_datetime_calib_format(dayList):

    """ Convert list of string dates into datetimes """

    datetimeDays = []

    for d in dayList:

        datetimeDays += [dt.datetime(int(d[0:4]), int(d[5:7]), int(d[8:10]))]

    return datetimeDays

if __name__ == '__main__':

    # ==============================================================================
    # Setup
    # ==============================================================================

    # directories
    maindir = 'C:/Users/Elliott/Documents/PhD Reading/LUMO - Sensor network/calibration/'
    datadir = maindir + 'data/'
    ceildatadir = datadir + 'L1/'
    perioddatadir = datadir + 'ceilometer_periods/'
    savedir = maindir + 'figures/'

    # site_bsc = {'CL31-B_RGS': 28.1 - 19.4, 'CL31-C_MR': 32.0 - 27.5, 'CL31-A_KSS45W': 64.3, 'CL31-D_NK': 27.0 - 23.2}
    site_ids = ['CL31-A_KSS45W', 'CL31-A_IMU', 'CL31-B_RGS', 'CL31-C_MR', 'CL31-D_NK', 'CL31-D_SWT', 'CL31-E_NK']

    # clear sky days to overplot onto thecalibration
    daystrList = ['20150414', '20150415', '20150421', '20150611', '20160504', '20160823', '20160911', '20161125',
                  '20161129', '20161130', '20161204']

    clear_days = eu.dateList_to_datetime(daystrList)

    # # regime styles - [start, end, regime type]
    # regimes = {'CL31-A_KSS45W': {'1': [dt.date(2015, 2, 24), dt.date(2015, 6, 20), 'time'],
    #                              '2': [dt.date(2015, 9, 01), dt.date(2016, 4, 01), 'block_avg']},
    #            'CL31-B_RGS': {'1': [dt.date(2015, 2, 5), dt.date(2016, 12, 31), 'window_transmission']},
    #            'CL31-C_MR': {'1': [dt.date(2015, 2, 5), dt.date(2016, 7, 28), 'block_avg'],
    #                          '2': [dt.date(2016, 7, 28), dt.date(2016, 12, 31), 'block_avg']},
    #            'CL31-D_NK': {'1': [dt.date(2015, 2, 5), dt.date(2016, 12, 31), 'window_transmission']}}

    # regime styles - [start, end, regime type]
    regimes = {'CL31-A_KSS45W': {'1': [dt.date(2015, 2, 24), dt.date(2015, 6, 20), 'time'],
                                 '2': [dt.date(2015, 9, 01), dt.date(2016, 4, 01), 'block_avg']},
               'CL31-B_RGS': {'1': [dt.date(2015, 2, 5), dt.date(2016, 12, 31), 'window_transmission']},
               'CL31-C_MR': {'1': [dt.date(2015, 2, 5), dt.date(2016, 7, 28), 'block_avg'],
                             '2': [dt.date(2016, 7, 28), dt.date(2016, 12, 31), 'block_avg']},
               'CL31-D_NK': {'1': [dt.date(2011, 6, 5), dt.date(2012, 5, 1), 'block_avg'],
                             '2': [dt.date(2015, 2, 5), dt.date(2016, 12, 31), 'window_transmission']}}

    calib_pro = {}

    years = [2016, 2017]

    # read L1 calibration data
    # calib_all = read_cal()

    # for each site
    for site_id in site_ids:

        # get site information from site_id
        ceil_id = site_id.split('_')[0]
        ceil_type = site_id.split('-')[0]
        site = site_id.split('_')[-1]

        print '... Processing ' + site

        # read in calibration data for site
        filepaths = [ceildatadir + ceil_id +'_CAL_'+site+'_'+str(i)+'.nc' for i in years]
        calib = eu.netCDF_read(filepaths)

        # this site calibration data
        # calib = calib_all[site]

        # ==============================================================================
        # Read data
        # ==============================================================================

        # read periods
        periods = read_periods(perioddatadir, site)

        # read transmission
        window_trans = read_window_trans(site, datadir)

        # read pulse energy
        pulse = read_pulse(site, datadir)

        # ==============================================================================
        # Process
        # ==============================================================================

        # turn dates into datetimes
        calib_dates = dateList_to_datetime_calib_format(calib['Dates'])
        calib_dates_days = np.array([i.date() for i in calib_dates])

        # linearly interpolat the nans
        # calibration = eu.linear_interpolation(data['wv_cal'])

        # mask raw data
        # calib_raw = np.ma.masked_where(np.isnan(calib['wv_cal']), calib['wv_cal'])

        # create  time series of dailymax window transmission
        window_trans_daily = calc_daily_window_trans(window_trans, calib_dates_days, calib)

        # remove event days from window transmission (firmware, hardware, cleaing etc)
        window_trans_daily = remove_events(periods, window_trans_daily)

        # process the calibration based on ceilometer 'regime' so there is data for all days.
        window_trans_daily = process_calibration_for_all_days(window_trans_daily, regimes, site)

        # save processed calibration data
        with open(perioddatadir + '/' + site + '_window_trans_daily_cpro.pickle', 'wb') as handle:
            pickle.dump(window_trans_daily, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # use this to open:
        # with open(perioddatadir + '/' + site + '_window_trans_daily_cpro.pickle', 'rb') as handle:
        #   window_trans_daily = pickle.load(handle)


        # # convert num samples to number of samples by hour (2 hours)
        # window_trans_daily['sample_hr'] = window_trans_daily['samples']/240.0
        #
        # # finding out what happened on those dates at the end of KSS45W
        # idx = np.where(window_trans_daily['c_wv'] > 3.5)
        # np.array(window_trans_daily['c_wv'])[np.where(window_trans_daily['c_wv'] > 3.5)]
        # np.where(np.array(window_trans_daily['dates']) > dt.date(2016, 3, 1))
        #
        #
        # mid_idx = np.where((np.array(window_trans_daily['dates']) > dt.date(2015, 2, 24)) &
        #          (np.array(window_trans_daily['dates']) < dt.date(2015, 8, 24)))
        # np.sum(~np.isnan(window_trans_daily['c_wv'][mid_idx]))
        #
        # # plt.scatter(np.array(window_trans_daily['dates'])[mid_idx], window_trans_daily['c_wv'][mid_idx])
        #
        # early_idx = np.where((np.array(window_trans_daily['dates']) < dt.date(2015, 4, 16)))
        # clip_early = {}
        # for key, value in window_trans_daily.iteritems():
        #     clip_early[key] = np.array(value)[early_idx]
        # # scatter_c_sample_period(clip, site, savedir, dt.date(2015, 2, 24), dt.date(2015, 8, 24))
        # scatter_window_c_period(clip_early, site, savedir, dt.date(2015, 2, 5), dt.date(2015, 4, 16))
        #
        # late_idx = np.where((np.array(window_trans_daily['dates']) > dt.date(2015, 4, 16)))
        # clip_late = {}
        # for key, value in window_trans_daily.iteritems():
        #     clip_late[key] = np.array(value)[late_idx]
        # # scatter_c_sample_period(clip, site, savedir, dt.date(2015, 2, 24), dt.date(2015, 8, 24))
        # scatter_window_c_period(clip_late, site, savedir, dt.date(2015, 4, 16), dt.date(2016, 12, 31))
        #
        # # d = [i.strftime('%Y%j') for i in np.array(window_trans_daily['dates'])[v]]
        # scatter_c_sample_period(window_trans_daily_period, site, savedir, startDay, endDay)




        # very quick plot histogram of sample size
        plot_hist(window_trans_daily)

        # plot data for each period separately
        plot_period(periods, window_trans_daily, site, savedir)
        # remove transmission enteries in periods
        # find all NONE transmission enteries
        # idx = np.where(np.array(periods['Type']) != 'Transmission')[0]










        # ----------------
        # moving average the data
        #ma_7 = eu.moving_average(calibration, 7)
        #ma_10 = eu.moving_average(calibration, 10)
        #ma_30 = eu.moving_average(calibration, 30)
        #
        # plot the smoothed data
        # plot_smooth(dates, calibration, ma_7, ma_10, ma_30)
        # ----------------

        # time-series point plot of calibration, window trans and pulse energy.
        # plot_cal_wind_pulse_timeseries(calib_dates, calib_raw, window_trans, pulse, savedir, site, clear_days)

        plot_cal_wind_pulse_timeseries_new_cal(window_trans_daily, savedir, site, clear_days)

        # scatter all the data for a site. Max daily window trans verses calibration coeff
        #scatter_window_c_all(window_trans_daily, site, savedir)

        #scatter_window_c_all_type(window_trans_daily, periods, site, savedir)

        plt.close('all')

    print 'END PROGRAM'