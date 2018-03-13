"""
Functions to support the ceilometer calibration script (LUMO_ceil_block_calibration.py)

Functions copied/tweaked from Emma Hopkin's set of Utility functions.

LUMO_calibration_Utils created by Elliott Thurs 08/03/2018

Original functions created by Emma Hopkin
Edited by Elliott Warren
"""

from netCDF4 import Dataset
import numpy as np
import datetime as dt

# Reading

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
    Read in an combine the two datasets, then time at the end.
    :param yest_filepath:
    :param day_filepath:
    :param day: the main data the calbration is for [datetime object]
    :param range_data:
    :param time_data:
    :param beta_data:
    :return:

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


    # trim data to ultimately keep data for just the main day - get idx to identify which data to keep from each
    #   need to make sure kept data does not overlap, hence yest_data will end where day_data takes over
    #   also include hour 24 of day, from day_data, as that was original used in the EH calibration code
    yest_idx = np.array([np.logical_and(i.date() == day.date(), i < day_data['pro_time'][0]) for i in yest_data['pro_time']])
    day_idx = np.array([np.logical_or(i.date() == day.date(), i == day + dt.timedelta(days=1)) for i in day_data['pro_time']])

    # merge the data
    data = {}
    for key in yest_data.iterkeys():
        if key == 'height':
            # just make a copy of it
            data['height'] = yest_data['height']
        else:
            # combine yest and day data together based on yest_idx and day_idx
            data[key] = np.concatenate((yest_data[key][yest_idx], day_data[key][day_idx]))


    # Equations from Ambaum, pg100
    e = (1.61 * np.array(data['q']) * np.array(data['pressure'])) / (1 + (1.61 * np.array(data['q'])) - np.array(data['q']))
    ukv_density2 = (0.0022 * e) / data['air_temperature']

    ####Interpolate model onto observation space#####
    WV_newgrid = np.zeros((25, (len(range_data.points))))
    for i in xrange(len(WV_newgrid)):
        WV_newgrid[i, :] = griddata(data['height'][:], ukv_density2[i, :], 1000.0 * range_data.points, method='linear')

    WV_obspace = np.zeros(np.shape(np.transpose(beta_data)))
    for j in xrange(len(np.transpose(WV_obspace))):
        WV_obspace[:, j] = griddata(ukv_time[:], WV_newgrid[:, j], time_data.points, method='linear')

    ####Calculate Trransmissivity####
    # for each WV profile, calculate cumulative integration then calculate transmission at each corresponding height.
    ukv_WV_Beta = np.zeros(np.shape(np.transpose(beta_data)))
    ukv_transmissivity = np.zeros(np.shape(np.transpose(beta_data)))
    ukv_integral = np.zeros(np.shape(np.transpose(beta_data)))

    for t in xrange(len(ukv_WV_Beta)):
        ukv_integral[t, :-1] = integrate.cumtrapz(WV_obspace[t, :], range_data.points[:] * 1000)
        ukv_transmissivity[t, :] = 1 - 0.17 * (ukv_integral[t, :] * 0.1) ** 0.52
        # set last gate to last but one - not included in integral due to array size matching
        ukv_transmissivity[t, -1] = ukv_transmissivity[t, -2]

    return (ukv_transmissivity)


# Processing

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