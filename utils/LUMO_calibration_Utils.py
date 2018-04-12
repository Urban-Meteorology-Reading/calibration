"""
Functions to support the ceilometer calibration script (LUMO_ceil_block_calibration.py)

Functions copied/tweaked from Emma Hopkin's set of Utility functions.

LUMO_calibration_Utils created by Elliott Thurs 08/03/2018
"""

from netCDF4 import Dataset
import numpy as np
import datetime as dt
import operator
import matplotlib.pyplot as plt
from scipy import stats


# Reading ------------------------------------------------------------

def mo_create_filenames(day, datadir_mo):

    """
    Create the filenames for the MO data to read in.
    :param day: [datetime]
    :param datadir_mo: main data directory for MO data, subdirectories from the date will be made in this function
    :return: yest_filepath: full filepath to yesterday's data
    :return: day_filepath: full filepath to day's data

    EW 08/03/18
    """

    # get yesterday's time too, for the MO NWP file read in
    yest = day - dt.timedelta(days=1)

    # get MO filepaths for the water vapour
    yesterday_filename = 'MOUKV_FC'+yest.strftime('%Y%m%d')+'06Z_WXT_KSSW.nc'
    day_filename = 'MOUKV_FC'+day.strftime('%Y%m%d')+'06Z_WXT_KSSW.nc'

    yest_filepath = datadir_mo + yesterday_filename
    day_filepath = datadir_mo + day_filename

    return yest_filepath, day_filepath

def mo_read_calc_wv_transmission(yest_filepath, day_filepath, day, range_data, time_data, beta_data):

    """
    Calculate the transmission given the presence of water vapour
    :param yest_filepath:
    :param day_filepath:
    :param day: the main data the calbration is for [datetime object]
    :param range_data: range (not height) [m]
    :param time_data:
    :param beta_data:
    :return: ukv_transmissivity

    Reads in water vapour data from NWP model data.

    EW 08/03/18
    """

    from scipy import integrate
    from scipy.interpolate import griddata
    import ellUtils as eu

    # read in data from a single file
    def read_single_mo_file(filepath):

        """
        Read in MO data from a single file
        :param filepath: single day filepath
        :return: data [dictionary] variables within it

        EW 09/03/18
        """

        # define dictionary
        data ={}

        # open files
        file = Dataset(filepath)

        # get time
        raw_start_time = file.variables['time'][:]
        tstr = file.variables['time'].units
        pro_start_time = eu.time_to_datetime(tstr, raw_start_time)[0]
        forecast_time = file.variables['forecast'][:]
        data['pro_time'] = np.array([pro_start_time + dt.timedelta(hours=int(i)) for i in forecast_time])

        # height
        data['height'] = file.variables['height'][:, 1, 1]

        # pressure [Pa] .shape(1, time, height, 3, 3)
        data['pressure'] = file.variables['P_rhoLev'][0, :, :, 1, 1]

        # temperature [K]
        data['air_temperature'] = file.variables['T_m'][0, :, :, 1, 1]

        # Specific humidity [kg kg-1]
        data['q'] = file.variables['QV'][0, :, :, 1, 1]

        # calculate air density
        data['dry_air_density'] = np.array(data['pressure']) / (Rstar * np.array(data['air_temperature']))
        data['wv_density'] = np.array(data['q']) * data['dry_air_density']

        return data

    # Gas constant for dry air
    Rstar = 287

    # read in and concatonate the two files
    yest_data = read_single_mo_file(yest_filepath)
    day_data = read_single_mo_file(day_filepath)


    # trim and merge data from both forecasts to get data for the main day - get idx to identify which data to keep for each forecast
    #   need to make sure kept data does not overlap, hence yest_data will end where day_data takes over
    #   also include hour 24 of day, from day_data, as that was original used in the EH calibration code
    #   need to consider the wind-up time of the second forecast - need to not include it and instead include more of the
    #       previous forecast.

    # hour to start for day and to go up to for yest.
    #   e.g. | yest -----> |[day, hr=forecast hour of day + 3 hours of wind up] day ---------> |
    day_forecast_start_hour = day_data['pro_time'][0].hour # forecast start time
    split_time = dt.datetime(day.year, day.month, day.day, day_forecast_start_hour + 3, day.minute, day.second)
    # get booleon arrays for each forecast
    yest_idx = np.array([split_time > i for i in yest_data['pro_time']])
    day_idx = np.array([split_time <= i for i in day_data['pro_time']])

    # day_idx = np.arange(19) # Emma's old idx
    # yest_idx = np.arange(31, len(yest_data['pro_time'])) # Emma's old idx

    # merge the data
    data = {}
    for key in yest_data.iterkeys():
        if key == 'height':
            # just make a copy of it
            data['height'] = yest_data['height']
        else:
            # combine yest and day data together based on yest_idx and day_idx
            data[key] = np.concatenate((yest_data[key][yest_idx], day_data[key][day_idx]))

    # get time as minutes since day start
    time_delta = time_data - day
    bsc_time_match = np.array([(i.total_seconds()) for i in time_delta])

    # seconds to interpolate onto
    time_delta = data['pro_time'] - day
    ukv_time = np.array([(i.total_seconds()) for i in time_delta])

    # Equations from Ambaum, pg100
    e = (1.61 * np.array(data['q']) * np.array(data['pressure'])) / (1 + (1.61 * np.array(data['q'])) - np.array(data['q']))
    ukv_density2 = (0.0022 * e) / data['air_temperature']

    ####Interpolate model onto observation space#####

    WV_newgrid = np.zeros((25, (len(range_data))))
    for i in xrange(len(WV_newgrid)):
        WV_newgrid[i, :] = griddata(data['height'], ukv_density2[i, :], range_data, method='linear')

    ####Interpolate model onto observation time#####
    WV_obspace = np.zeros(np.shape(np.transpose(beta_data)))
    for j in xrange(len(np.transpose(WV_obspace))):
        WV_obspace[:, j] = griddata(ukv_time, WV_newgrid[:, j], bsc_time_match, method='linear')

    ####Calculate Trransmissivity####
    # for each WV profile, calculate cumulative integration then calculate transmission at each corresponding height.
    ukv_WV_Beta = np.zeros(np.shape(np.transpose(beta_data)))
    ukv_transmissivity = np.zeros(np.shape(np.transpose(beta_data)))
    ukv_integral = np.zeros(np.shape(np.transpose(beta_data)))

    for t in xrange(len(ukv_WV_Beta)):
        ukv_integral[t, :-1] = integrate.cumtrapz(WV_obspace[t, :], range_data)
        ukv_transmissivity[t, :] = 1 - 0.17 * (ukv_integral[t, :] * 0.1) ** 0.52
        # set last gate to last but one - not included in integral due to array size matching
        ukv_transmissivity[t, -1] = ukv_transmissivity[t, -2]

    return (ukv_transmissivity)


# Processing ------------------------------------------------------------

def find_cloud(beta_data):

    """
    Find the location of the max beta
    Take cloud range as 15 gates above and below
    return array with 30gate range beta only; others set to nan
    :param beta_data:
    :return:

    EH
    """

    arr = np.copy(beta_data)
    for i in xrange(len(np.transpose(arr))):

        index, value = max(enumerate(arr[:, i]), key=operator.itemgetter(1))
        loc_above = index + 15
        loc_below = index - 15
        if loc_below < 0:
            loc_below = 0
        else:
            pass
        arr[loc_above:, i] = np.nan  # set everything above to nan
        arr[:loc_below, i] = np.nan  # set everything below to nan
    return (arr)

def scatter_correct_Vais(cloud_beta, range_km):

    """
    Input: raw beta data
    Output: beta with scattering correxction applied
    Apply multiple scattering correction - source:  http://www.met.reading.ac.uk/~swr99ejo/lidar_calibration/index.html

    :param cloud_beta: raw backscatter data .shape(height, time)
    :param range_km: ceilometer range gates in [km]
    :return: Scat_correct_b: corrected cloud_beta backscatter

    EH
    Edited by EW 09/03/18
    """

    # ToDo - Ask EH if range_km should be a range or height for the backscatter

    Scat_correct_b = np.copy(cloud_beta)

    #Scat_correct_b = 1.6*(beta_data)
    #Apply height dependent eta to beta values - i.e. multiply out multiple scattering
    #Note these values are instrument dependent
    ind1 = np.where(range_km< 0.250)
    Scat_correct_b[ind1,:]=Scat_correct_b[ind1,:]*0.82854
    ind2 = np.where((np.abs(range_km - 0.375)) < 0.125)
    Scat_correct_b[ind2,:]=Scat_correct_b[ind2,:]*0.82371
    ind3 = np.where((np.abs(range_km - 0.625)) < 0.125)
    Scat_correct_b[ind3,:]=Scat_correct_b[ind3,:]*0.81608
    ind4 = np.where((np.abs(range_km - 0.875)) < 0.125)
    Scat_correct_b[ind4,:]=Scat_correct_b[ind4,:]*0.80811
    ind5 = np.where((np.abs(range_km - 1.125)) < 0.125)
    Scat_correct_b[ind5,:]=Scat_correct_b[ind5,:]*0.79969
    ind6 = np.where((np.abs(range_km - 1.375)) < 0.125)
    Scat_correct_b[ind6,:]=Scat_correct_b[ind6,:]*0.79027
    ind7 = np.where((np.abs(range_km - 1.625)) < 0.125)
    Scat_correct_b[ind7,:]=Scat_correct_b[ind7,:]*0.78227
    ind8 = np.where((np.abs(range_km - 1.875)) < 0.125)
    Scat_correct_b[ind8,:]=Scat_correct_b[ind8,:]*0.77480
    ind9 = np.where((np.abs(range_km - 2.125)) < 0.125)
    Scat_correct_b[ind9,:]=Scat_correct_b[ind9,:]*0.76710
    ind10 = np.where((np.abs(range_km - 2.375)) < 0.125)
    Scat_correct_b[ind10,:]=Scat_correct_b[ind10,:]*0.76088

    return (Scat_correct_b)

def corr_beta(Scat_beta, beta_data):
    """
    # MERGE SCATTERING CORRECTION WITH BETA DATA
    Locate nans placed by finding cloud(see above)
    replace with beta values
    s_cor_beta is array of beta with scatting correction applied to 30 gates around max beta
    """
    s_cor_beta = np.copy(Scat_beta)

    for prof in xrange(len(np.transpose(s_cor_beta))):
        index_n = np.isnan(Scat_beta[:, prof])
        thenans = np.where(index_n == True)
        for locnan in xrange(len(thenans)):
            s_cor_beta[(thenans[locnan]), prof] = beta_data[(thenans[locnan]), prof]

    return (s_cor_beta)

def lidar_ratio(Scat_correct_b, range_data):

    """
    Input:
    Output:

    :param Scat_correct_b: scatter corrected beta data (attenuated backscatter coefficient)
    :param range_data:
    :return: S, S2: lidar ratio (ratio backscatter coefficient to extinction coefficient)

    """

    gatesize = np.round((range_data[13]*1000 - range_data[12]*1000),3) #km to m
    print 'GATESIZE = ', gatesize
    GateMax = (np.where(range_data > 2400/1000.)[0][0])
    begin = (np.where(range_data > 10/1000.)[0][0])

    # Integrated from 0.2 - 2400km
    inc_beta = Scat_correct_b[begin:GateMax,:]      #betas between 0.2 - 2400km
    S = []
    S2 = []
    for i in xrange(len(np.transpose(inc_beta))):
        peak, value = max(enumerate(inc_beta[:,i]), key=operator.itemgetter(1))
        B_aer_prof = ((sum(inc_beta[:(peak-8),i]))*gatesize)
        B_cloud_prof = ((sum(inc_beta[(peak-8):,i]))*gatesize)
        B_tot = 2*B_aer_prof + B_cloud_prof

        integrate_S = ((sum(inc_beta[:,i]))*gatesize)
        #print 'Without T: ',integrate_S
        #integrate_S = integrate_S * transmittance(50.,B_aer_prof)
        #print 'With T: ', integrate_S
        prof_S = ((integrate_S*2)**-1)
        prof_tot_S = ((B_tot*2)**-1)
        S.append(prof_S)
        S2.append(prof_tot_S)

    S = list(S)
    S2 = list(S2)

    return(S, S2)

## Filtering

def step1_filter(beta_data, range_data, maxB_val, ratio_val, S):

    """
    Using scatter corrected data, run through several filters to remove profiles unsuitable for calibration
    filter1: aerosol from 100m below CBH (max B) to gate5 must not be more than (e.g).5% of total integrated B (0.2-2.4km)
    filter2: B 300m above max B must be 20x smaller
    filter3: B 300m below max B must be 20x smaller
    filter4: Log Max B must be at least -5 m-1 sr-1
    NB. filter functions above, called by this one
    beta data with nans for profiles that did not pass the filters
    OUTPUT: S with nan where filters not passed

    :param beta_data:
    :param range_data:
    :param maxB_val:
    :param ratio_val:
    :param S:
    :return:

    EH before 15/03/18
    """

    def filter_ratio(beta_data, range_data, ratio_val):

        """
        Hogan et al (2003b); O'Connor et al (2004) - exclude strong background aerosol events
        Input: raw beta data, set required ratio
        integrate over full profile values and integrate up to 100m below CBH
        If beta below CBH represents more than 5% of total, profile discarded
        Output: beta with nans for profiles that don't qualify

        :param beta_data:
        :param range_data:
        :param ratio_val:
        :return: filtered_S
        :return: B_ratio
        """

        def total_beta(beta_data, range_data):

            """

            :param beta_data:  scatter corrected backscatter data, (between selected indices??)
            :param range_data:
            :return: integrate_beta: integrated beta for each 30s profile

            EH before 15/03/18
            """

            # Gate size dependent on instrument
            GateMax = (np.where(range_data > 2400 / 1000.)[0][0])
            gatesize = np.round((range_data[13] * 1000 - range_data[12] * 1000), 3)

            inc_beta = beta_data[5:GateMax, :]  # betas above first 5 gates (near view issues)
            integrate_beta = ((sum(inc_beta)) * gatesize)
            return (integrate_beta)

        def inteB_belowCBH(beta_data, range_data):

            """
            Integrate beta up to 100m below max beta
            ???Where CBH is below 200m range, beta is returned as zero - changed to nan

            :param beta_data:
            :param range_data:
            :return:

            EH before 15/03/18
            """

            # Gate size dependent on instrument
            G100m = (np.where(range_data > 100 / 1000.)[0][0])
            gatesize = np.round((range_data[13] * 1000 - range_data[12] * 1000), 3)

            inte_beta = []
            a = len(np.transpose(beta_data))
            for prof in xrange(a):
                loc_belowCBH = np.where(beta_data[:, prof] == np.max(beta_data[:, prof]))[0][0]
                loc = loc_belowCBH - G100m  # 100m below

                inc_beta = beta_data[5:loc, prof]  # betas between gate 5 - 100m below
                integrate_beta = ((sum(inc_beta)) * gatesize)
                inte_beta.append(integrate_beta)
            inte_beta[inte_beta == 0] = np.nan

            return (inte_beta)

        integrated_B = total_beta(beta_data, range_data)
        integrated_belCBH_B = inteB_belowCBH(beta_data, range_data)
        B_ratio = integrated_belCBH_B / integrated_B
        # B_ratio = beta_ratio.tolist()
        ratiof_B = np.copy(beta_data)
        filt_out = 0

        # eliminate profiles where beta below CBH represents more than 5% of total
        for i in range(len(B_ratio)):
            if B_ratio[i] > ratio_val:
                ratiof_B[:, i] = np.nan
                filt_out = filt_out + 1
            elif B_ratio[i] < 0:
                ratiof_B[:, i] = np.nan
                filt_out = filt_out + 1
            else:
                pass
        print 'ratio filtered out: ', filt_out

        return (ratiof_B, B_ratio)

    def filter_300(beta_data, range_data):

        """
        Hogan et al (2003b); O'Connor et al (2004) - to be a thick highly attenuating cloud, peak att_Beta must be a factor of 20 times greater than 300m above
        Instrument dependent - different gate sizes
        Where the condition is not passed, value is replaced with a NaN

        :param beta_data:
        :param range_data:
        :return: beta_300f
        """

        # Gate size dependent on instrument
        # Ensure cloud is above 100m
        Gate300m = (np.where(range_data > 350 / 1000.)[0][0]) - (
            np.where(range_data > 50 / 1000.)[0][0])

        beta_300f = np.copy(beta_data)
        a = np.zeros(Gate300m)
        a = np.transpose(a)
        ex2 = 0
        import operator

        for i in xrange(len(np.transpose(beta_300f))):
            profs_sort = np.append(beta_300f[:, i], a)

            index, value = max(enumerate(beta_300f[:, i]), key=operator.itemgetter(1))
            loc_compare_p = index + Gate300m
            compare_p = profs_sort[(loc_compare_p)]

            if loc_compare_p < 0 or compare_p * 20 > value:
                beta_300f[:, i] = np.nan
                ex2 = ex2 + 1
            else:
                pass

        print'300m above filtered out: ', ex2

        return (beta_300f)

    def filter_300below(beta_data, range_data):

        """
        Follows reasoning of 300m above filtering
        Instrument dependent - different gate sizes
        Where the condition is not passed, value is replaced with a NaN

        :param beta_data:
        :param range_data:
        :param Instrument:
        :return:
        """

        Gate300m = (np.where(range_data > 350 / 1000.)[0][0]) - (np.where(range_data > 50 / 1000.)[0][0])

        beta_300belf = np.copy(beta_data)
        # add on 300m of zeros so bottom of profile isnt compared to top
        a = np.zeros(Gate300m)
        a = np.transpose(a)
        ex2 = 0
        import operator

        for i in xrange(len(np.transpose(beta_300belf))):
            profs_sort = np.append(beta_300belf[:, i], a)

            index, value = max(enumerate(beta_300belf[:, i]), key=operator.itemgetter(1))
            loc_compare_p = index - Gate300m
            compare_p = profs_sort[(loc_compare_p)]

            if loc_compare_p < 0 or compare_p * 20 > value:
                beta_300belf[:, i] = np.nan
                ex2 = ex2 + 1
            else:
                pass
        print '300m below filtered out: ', ex2

        return (beta_300belf)

    def filter_maxB(beta_data, threshold):  # run either with original beta or beta_300f

        """
        Hogan et al (2003b); O'Connor et al (2004) - peak att_Beta should be more than 10-4 sr-1 m-1
        Filtered loosened to 10-5 sr-1 m-1 to allow for uncalibrated instruments
        Input: beta array
        Output: Max values of beta (btwn 1.0 and 2.4km), beta filtered with log max > -5

        :param beta_data:
        :param threshold:
        :return:
        """

        beta_maxf = np.copy(beta_data)
        log_beta = np.log10(beta_maxf)
        log_beta[np.isnan(log_beta)] = -100  # remove nan and inf value
        log_beta[np.isinf(log_beta)] = -100  # -100 not representative of backscatter noise

        ex = 0
        maxs = np.zeros(len(np.transpose(beta_maxf)))
        for i in xrange(len(np.transpose(beta_maxf))):
            maxs[i] = np.max(beta_maxf[:, i])
            if np.max(log_beta[:, i]) < threshold:
                beta_maxf[:, i] = np.nan
                ex = ex + 1
            else:
                pass
        print 'MaxB filtered out: ', ex

        return (maxs, beta_maxf)

    def locate_nan(arr):

        """
        Input: version of beta with nans created by a filter
        Output: location of nans in beta array
        """
        index_n = np.isnan(arr)  # e.g. beta_300f
        find_n = np.where(index_n == True)
        loc_n1 = find_n[1]
        loc_n2 = np.squeeze(loc_n1)
        loc_n = np.unique(loc_n2)

        return (loc_n)

    def filter_bynan(loc_n, S):

        """
        Input: Output for locate_nan() and second array to filter (usually S)
        Output: Filtered version of e.g. S (nans in place)
        """

        print np.shape(S)
        filt_S = np.copy(S)
        for loc in range(len(loc_n)):
            filt_S[(loc_n[loc])] = np.nan

        return (filt_S)

    filter1, B_ratio = filter_ratio(beta_data, range_data, ratio_val)
    filter2 = filter_300(filter1, range_data)
    filter3 = filter_300below(filter2, range_data)
    maxBs, filter4 = filter_maxB(filter3, maxB_val)

    loc_beta_f = locate_nan(filter4)
    filtered_S = filter_bynan(loc_beta_f, S)
    return (filtered_S, B_ratio)

def step2_Sfilt(S, S_range_percent, no_of_profs):

    """
    New version of step2_Sfilt which requires a set number profiles in a row to be
    within 10% of the mean of those profiles (low variability). Steps forward one profile
    at a time. If the profiles meet the criteria, all 7 are copied, else they
    are left as zeros. All remaining zeros converted to nans

    :param S:
    :param S_range_percent:
    :param no_of_profs:
    :return:
    """

    plus = 100. + S_range_percent
    minus = 100. - S_range_percent
    const_S = np.zeros(len(S))

    for i in range(len(S) - (no_of_profs - 1)):
        group = S[i:i + (no_of_profs)]

        mean_of_group = np.mean(group)
        range_check = group / mean_of_group * 100
        if (np.min(range_check) > minus and np.max(range_check) < plus):
            const_S[i:i + (no_of_profs)] = S[i:i + (no_of_profs)]
        else:
            pass

    const_S[const_S == 0.] = np.nan

    return (const_S)

## Statistics

def S_mode_mean(Step2_S, Cal_hist):
    """
    Calculate daily mode for S
    if number of profiles in mean is less than 20 (10 mins), daily stats will not be used in the calibration (ie are set to nan) - this is derived from Cal_hist
    Cal_hist gives histogram values so max gives the mode
    """
    var = np.copy(Step2_S)
    peak = np.max(Cal_hist)
    if peak > 10.:
        print '@@@@@@@@', peak, '@@@@@@@'
        m = np.asarray(var)

        m[np.isnan(m)] = 0
        m = m[np.nonzero(m)]
        mean = np.mean(m)  ###mean
        print 'calibration mean = ', mean

        stdev = np.std(m)
        print 'std deviation = ', stdev

        sem = stats.sem(m)
        print 'std error = ', sem

        m2 = np.round((m.tolist()), 1)
        mode_arr = stats.mode(m2)
        mode = mode_arr[0][0]
        print 'calibration mode = ', mode

        median = np.median(m)
        print 'calibration median = ', median

        C_data = m / 18.8

        C_median = np.median(C_data)
        print 'C median = ', C_median

        C_mode_arr = stats.mode(m2 / 18.8)
        C_mode = C_mode_arr[0][0]
        print 'C mode = ', C_mode

        C_stdev = np.std(C_data)
        print 'C std = ', C_stdev

        CL_data = 1. / (m / 18.8)

        CL_median = np.median(CL_data)
        print 'CL median = ', CL_median

        CL_stdev = np.std(CL_data)
        print 'CL std = ', CL_stdev

    else:
        print '######', peak, '#####'
        mode = np.nan
        mean = np.nan
        median = np.nan
        sem = np.nan
        stdev = np.nan
        C_mode = np.nan
        C_median = np.nan
        C_stdev = np.nan
        CL_median = np.nan
        CL_stdev = np.nan

    return (peak, mode, mean, median, sem, stdev, C_mode, C_median, C_stdev, CL_median, CL_stdev)

# Plotting -------------------------------------------------------------

def Plot_ashist(variable):
    v = np.copy(variable)
    v[np.isnan(v)] = 0
    vartoplot = v[np.nonzero(v)]
    if len(vartoplot) > 2:
        b = (np.round(np.max(vartoplot)))

        # plt.figure(figsize = (6,4))
        counts, bins, range = plt.hist(vartoplot, bins=(2 * b), range=(0, b))
        # # plt.xticks(np.arange(0, 110, 10.0), fontsize = 18)
        # plt.yticks(fontsize=18)
        # # plt.axis(xmax = b+1, xmin = 16)
        # bin_centers = 0.5 * np.diff(bins) + bins[:-1]
        #
        # plt.title('S', fontsize=20)
        # plt.xlabel('Apparent S [sr]', fontsize=18)
        # plt.ylabel('Frequency', fontsize=18)
        # plt.tight_layout()
        #
        # # plt.axis(xmax = 100)
        # plt.show()
    else:
        counts = 0
    return (counts, len(vartoplot))