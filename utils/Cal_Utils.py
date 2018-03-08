"""
Emma Hopkin
11th May 2015
Functions calculate lidar ratio and to filter beta and S data
"""
import os # operating system library to issue Unix commands
from os.path import join # use join to link file names to the directory (avoids error with '/' in the path)
#import ceilometer_utils as cutils, cristina's utilities. you won't need this yet.
import iris  # data management and plotting library from met office
import ncUtils # library for reading netCDF files
import iris.quickplot as qplt # more plotting routines
import matplotlib.pyplot as plt # plotting library (not Met Office)
from netCDF4 import Dataset # to read standard netCDF files
import numpy as np # numerical python library for arrays and mathematical functions
import copy
import operator
from scipy import stats
import scipy.io
from scipy.interpolate import interp1d
#       ###########################
#       ##                       ##
#       ## Calculate Lidar Ratio ##
#       ##                       ##
#       ###########################
def int_profile(Scat_correct_b,range_data, Instrument):
    """
    Input:scatter corrected beta data (attenuated backscatter coefficient)
    Output: integration of profile which would be used to calculate S
    """
    gatesize = np.round((range_data.points[13]*1000 - range_data.points[12]*1000),3) #km to m  
    print 'GATESIZE = ', gatesize   
    # Gate size dependent on instrument
    
    if Instrument == 'Vais':
        GateMax = 100
        #gatesize = 20
        begin = 25
    elif Instrument == 'Jen':
        GateMax = 266 #4000m
        gatesize = 15 
        begin = 33 #500m
    elif Instrument == 'CL51':
        GateMax = 400#4000m
        #gatesize = 10
        begin = 20
    elif Instrument == 'SIRTA':
        GateMax = 200 #3000m
        #gatesize = 15
        begin = 13 #~200m
    elif Instrument == 'LUMO':
        GateMax = 400 #4000m
        #gatesize = 10
        begin = 20        
    else:
        print 'Incorrect Instrument Input'    
    # Integrated from 0.2 - 2400km 
    inc_beta = Scat_correct_b[begin:GateMax,:]      #betas between 0.2 - 2400km
    integrate_S = ((sum(inc_beta))*gatesize)
    return(integrate_S)
    
    
def lidar_ratio_withT2(Scat_correct_b,range_data, Instrument, aer_S):
    """
    Input:scatter corrected beta data (attenuated backscatter coefficient)
    Output: lidar ratio (ratio backscatter coefficient to extinction coefficient)
    """
    gatesize = np.round((range_data.points[13]*1000 - range_data.points[12]*1000),3) #km to m  
    print 'GATESIZE = ', gatesize
    GateMax = (np.where(range_data.points > 2400/1000.)[0][0])
    begin = (np.where(range_data.points > 10/1000.)[0][0])
    # Gate size dependent on instrument
    #if Instrument == 'Vais':
    #    GateMax = 100
    #    #gatesize = 20
    #    begin = 25
    #elif Instrument == 'Jen':
    #    GateMax = 266
    #    #gatesize = 15 
    #    begin = 33 #500m 
    #elif Instrument == 'CL51':
    #    GateMax = 400
    #    #gatesize = 10
    #    begin = 20
    #elif Instrument == 'SIRTA':
    #    GateMax = 200 #3000m
        #gatesize = 15
    #    begin = 13 #~200m   
    #elif Instrument == 'LUMO':
    #    GateMax = 400 #4000m
        #gatesize = 10
    #    begin = 20             
    #else:
    #    print 'Incorrect Instrument Input'    
    # Integrated from 0.2 - 2400km 
    inc_beta = Scat_correct_b[begin:GateMax,:]      #betas between 0.2 - 2400km
    S = []
    B_T2 = []
    for i in xrange(len(np.transpose(inc_beta))):
        peak, value = max(enumerate(Scat_correct_b[:,i]), key=operator.itemgetter(1))#find peak of backscatter (cloud)

        B_aer_prof = ((sum(Scat_correct_b[:(peak-5),i]))*gatesize)#integrate upto just below cloud (aerosol)

        integrate_S = ((sum(inc_beta[:,i]))*gatesize)#integral for S (excludes problem gates below 500m)
        if i == 259:
            print 'Without T: ',integrate_S
        aer_transmittance = transmittance(aer_S,B_aer_prof)#calculate transmission correction using all of profile below cloud
        if i == 259:
            print 'transmittance correction: ', aer_transmittance
        integrate_S2 = integrate_S / aer_transmittance
        if i == 259:
            print 'With T: ', integrate_S2
        prof_S = ((integrate_S2*2)**-1)
        S.append(prof_S)
        B_T2.append(integrate_S2)
    S = list(S)    
    return(S, B_T2)
    
def lidar_ratio(Scat_correct_b,range_data, Instrument):
    """
    Input:scatter corrected beta data (attenuated backscatter coefficient)
    Output: lidar ratio (ratio backscatter coefficient to extinction coefficient)
    """
    gatesize = np.round((range_data.points[13]*1000 - range_data.points[12]*1000),3) #km to m  
    print 'GATESIZE = ', gatesize
    GateMax = (np.where(range_data.points > 2400/1000.)[0][0])
    begin = (np.where(range_data.points > 10/1000.)[0][0])
    # Gate size dependent on instrument
    #if Instrument == 'Vais':
    #    GateMax = 100
    #    #gatesize = 20
    #    begin = 25
    #elif Instrument == 'Jen':
    #    GateMax = 266
    #    #gatesize = 15 
    #    begin = 33 #500m 
    #elif Instrument == 'CL51':
    #    GateMax = 400
    #    #gatesize = 10
    #    begin = 20
    #elif Instrument == 'SIRTA':
    #    GateMax = 200 #3000m
        #gatesize = 15
    #    begin = 13 #~200m   
    #elif Instrument == 'LUMO':
    #    GateMax = 400 #4000m
        #gatesize = 10
    #    begin = 20             
    #else:
    #    print 'Incorrect Instrument Input'    
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
    
def transmittance(aer_S, B_prof):
    #print 'S', aer_S
    #print B_prof
    ex_prof = aer_S * B_prof
    #print ex_prof
    trans2 = (np.exp(-2*ex_prof))
    return (trans2)
###########################################################################
#           ######   Attenuated Beta Filters   ##########
###########################################################################
#include only profiles where beta 300m above max is at least 20 times smaller
def filter_300 (beta_data,range_data, Instrument ):
    """
    Hogan et al (2003b); O'Connor et al (2004) - to be a thick highly attenuating cloud, peak att_Beta must be a factor of 20 times greater than 300m above
    Instrument dependent - different gate sizes
    Where the condition is not passed, value is replaced with a NaN
    """
    # Gate size dependent on instrument
    # Ensure cloud is above 100m
    Gate300m = (np.where(range_data.points > 350/1000.)[0][0]) - (np.where(range_data.points > 50/1000.)[0][0])
    
    #if Instrument == 'Vais':
    #    Gate300m = 15
    #elif Instrument == 'Jen':
    #    Gate300m = 20
    #elif Instrument == 'CL51':
    #    Gate300m = 30   
    #elif Instrument == 'SIRTA':
    #    Gate300m = 20 
    #elif Instrument == 'LUMO':
    #    Gate300m = 30             
    #else:
    #    print 'Incorrect Instrument Input'
        
    beta_300f = np.copy(beta_data)
    a = np.zeros(Gate300m)
    a = np.transpose(a)
    ex2 = 0
    import operator
    
    for i in xrange (len(np.transpose(beta_300f))):
        profs_sort = np.append(beta_300f[:,i],a)

        index, value = max(enumerate(beta_300f[:,i]), key=operator.itemgetter(1))
        loc_compare_p = index + Gate300m
        compare_p = profs_sort[(loc_compare_p)]
        
        if loc_compare_p < 0 or compare_p*20 > value: 
            beta_300f[:,i] = np.nan
            ex2 = ex2 +1
        else:
            pass

    print'300m above filtered out: ', ex2 
    
    return (beta_300f)
#------------------------------------------------------------------------#
#include only profiles where beta 300m below max is at least 20 times smaller
def filter_300below (beta_data,range_data,Instrument):
    """
    Follows reasoning of 300m above filtering
    Instrument dependent - different gate sizes
    Where the condition is not passed, value is replaced with a NaN
    """
    Gate300m = (np.where(range_data.points > 350/1000.)[0][0]) - (np.where(range_data.points > 50/1000.)[0][0])
    # Gate size dependent on instrument
    # Ensure cloud is above 100m
    #if Instrument == 'Vais':
    #    Gate300m = 15 
    #elif Instrument == 'Jen':
    #    Gate300m = 20
    #elif Instrument == 'CL51':
    #    Gate300m = 30
    #elif Instrument == 'SIRTA':
    #    Gate300m = 20 
    #elif Instrument == 'LUMO':
    #    Gate300m = 30             
    #else:
    #    print 'Incorrect Instrument Input'
        
    beta_300belf = np.copy(beta_data)
    #add on 300m of zeros so bottom of profile isnt compared to top
    a = np.zeros(Gate300m)
    a = np.transpose(a)
    ex2 = 0
    import operator
    
    for i in xrange (len(np.transpose(beta_300belf))):
        profs_sort = np.append(beta_300belf[:,i],a)

        index, value = max(enumerate(beta_300belf[:,i]), key=operator.itemgetter(1))
        loc_compare_p = index - Gate300m
        compare_p = profs_sort[(loc_compare_p)]
        
        if loc_compare_p < 0 or compare_p*20 > value: 
            beta_300belf[:,i] = np.nan
            ex2 = ex2 +1
        else:
            pass
    print '300m below filtered out: ', ex2 
    
    return (beta_300belf)    
     
#---------------------------------------------------------------------------------------#
def filter_maxB (beta_data, threshold):                #run either with original beta or beta_300f
    """
    Hogan et al (2003b); O'Connor et al (2004) - peak att_Beta should be more than 10-4 sr-1 m-1
    Filtered loosened to 10-5 sr-1 m-1 to allow for uncalibrated instruments
    Input: beta array
    Output: Max values of beta (btwn 1.0 and 2.4km), beta filtered with log max > -5
    """
    beta_maxf = np.copy(beta_data)
    log_beta = np.log10(beta_maxf)
    log_beta[np.isnan(log_beta)] = -100     #remove nan and inf value 
    log_beta[np.isinf(log_beta)] = -100     # -100 not representative of backscatter noise 
    
    ex = 0
    maxs = np.zeros(len(np.transpose(beta_maxf)))
    for i in xrange (len(np.transpose(beta_maxf))):
        maxs[i] = np.max(beta_maxf[:,i])
        if np.max(log_beta[:,i]) < threshold:
            beta_maxf[:,i] = np.nan
            ex = ex+1
        else:
            pass
    print 'MaxB filtered out: ', ex
        
        
    return (maxs, beta_maxf)
#------------------------------------------------------------------------# 
def total_beta(beta_data, range_data, Instrument):
    """
    Input: scatter corrected beta data from cube, (between selected indices??)
    Output: integrated beta for each 30s profile
    """
    # Gate size dependent on instrument
    GateMax = (np.where(range_data.points > 2400/1000.)[0][0])
    gatesize = np.round((range_data.points[13]*1000 - range_data.points[12]*1000),3) 
    #if Instrument == 'Vais':
    #    GateMax = 100
    #    gatesize = 20
    #elif Instrument == 'Jen':
    #    GateMax = 266 #4km
    #    gatesize = 15
    #elif Instrument == 'CL51':
    #    GateMax = 400
    #    gatesize = 10 
    #elif Instrument == 'SIRTA':
    #    GateMax = 200
    #    gatesize = 15
    #elif Instrument == 'LUMO':
    #    GateMax = 400
    #    gatesize = 10                   
    #else:
    #    print 'Incorrect Instrument Input'    
    
    inc_beta = beta_data[5:GateMax,:]   #betas above first 5 gates (near view issues)
    integrate_beta = ((sum(inc_beta))*gatesize)
    return (integrate_beta)


def inteB_belowCBH(beta_data,range_data, Instrument):
    """
    Integrate beta up to 100m below max beta 
    ???Where CBH is below 200m range, beta is returned as zero - changed to nan
    """
    # Gate size dependent on instrument
    G100m = (np.where(range_data.points > 100/1000.)[0][0])
    gatesize = np.round((range_data.points[13]*1000 - range_data.points[12]*1000),3) 
    #if Instrument == 'Vais':
    #    G100m = 5
    #    gatesize = 20
    #elif Instrument == 'Jen':
    #    G100m = 7 ##rounded up##
    #    gatesize = 15
    #elif Instrument == 'CL51':
    #    G100m = 10
    #    gatesize = 10     
    #elif Instrument == 'SIRTA':
    #    G100m = 7
    #    gatesize = 15 
    #elif Instrument == 'LUMO':
    #    G100m = 10
    #    gatesize = 10                
    #else:
    #    print 'Incorrect Instrument Input' 
    # Sum beta_data values of profile
    inte_beta = []
    a = len(np.transpose(beta_data))
    for prof in xrange(a):
        loc_belowCBH = np.where(beta_data[:,prof] == np.max(beta_data[:,prof]))[0][0]
        loc = loc_belowCBH - G100m #100m below

        inc_beta = beta_data[5:loc,prof]      #betas between gate 5 - 100m below
        integrate_beta = ((sum(inc_beta))*gatesize)
        inte_beta.append(integrate_beta)
    inte_beta[inte_beta==0] = np.nan
    return(inte_beta)
    

def filter_ratio(beta_data,range_data, ratio_val, Instrument):
    """
    Hogan et al (2003b); O'Connor et al (2004)    - exclude strong background aerosol events 
    Input: raw beta data, set required ratio
    integrate over full profile values and integrate up to 100m below CBH
    If beta below CBH represents more than 5% of total, profile discarded
    Output: beta with nans for profiles that don't qualify
    """
    integrated_B = total_beta(beta_data, range_data, Instrument)
    integrated_belCBH_B = inteB_belowCBH(beta_data,range_data, Instrument)
    B_ratio = integrated_belCBH_B/integrated_B
    #B_ratio = beta_ratio.tolist()
    ratiof_B = np.copy(beta_data)
    filt_out = 0
    #eliminate profiles where beta below CBH represents more than 5% of total
    for i in range(len(B_ratio)):
        if B_ratio[i] > ratio_val:
            ratiof_B[:,i] = np.nan
            filt_out = filt_out +1
        elif B_ratio[i] < 0:
            ratiof_B[:,i] = np.nan 
            filt_out = filt_out +1       
        else:
            pass 
    print 'ratio filtered out: ', filt_out   
    return(ratiof_B, B_ratio)
    
#--------------------------------------------------------------------------#   
#filter by window
def filter_window(beta_data, daycube, window_crit):
    win_tau = daycube.coord('window_tau, % unobscured')
    beta_win = np.copy(beta_data)
    for i in xrange (len(np.transpose(beta_data))):
        if win_tau.points[i] < window_crit:
            #print '****************yes************'
            beta_win[:,i] = np.nan
        else:
            pass
    return(beta_win)
 #filter by power
def filter_power(beta_data, daycube, power_crit):
    pulse_energy = daycube.coord('Laser pulse energy, % of nominal factory setting')
    beta_pul = np.copy(beta_data)
    for i in xrange (len(np.transpose(beta_data))):
        if pulse_energy.points[i] < power_crit:
            #print '****************yes************'
            beta_pul[:,i] = np.nan
        else:
            pass
    return(beta_pul)   
############### Bring att_beta filtering together######################
def locate_nan(arr):
    """
    Input: version of beta with nans created by a filter
    Output: location of nans in beta array
    """
    index_n = np.isnan(arr)                 #e.g. beta_300f
    find_n = np.where(index_n == True)
    loc_n1 = find_n[1]
    loc_n2 = np.squeeze(loc_n1)
    loc_n = np.unique(loc_n2)
    return (loc_n)
########    
def filter_bynan (loc_n,S):
    """
    Input: Output for locate_nan() and second array to filter (usually S)
    Output: Filtered version of e.g. S (nans in place)
    """
    print np.shape(S)
    filt_S = np.copy(S)
    for loc in range(len(loc_n)):
        filt_S[(loc_n[loc])] = np.nan
    return (filt_S)
########
def step1_filter(beta_data, range_data,maxB_val,ratio_val,S, Instrument):
    """
    Using scatter corrected data, run through several filters to remove profiles unsuitable for calibration
    filter1: aerosol from 100m below CBH (max B) to gate5 must not be more than (e.g).5% of total integrated B (0.2-2.4km)    
    filter2: B 300m above max B must be 20x smaller
    filter3: B 300m below max B must be 20x smaller 
    filter4: Log Max B must be at least -5 m-1 sr-1
    NB. filter functions above, called by this one
    beta data with nans for profiles that did not pass the filters
    OUTPUT: S with nan where filters not passed
    """
    filter1, B_ratio = filter_ratio(beta_data, range_data, ratio_val, Instrument)    
    filter2 = filter_300 (filter1, range_data, Instrument)
    filter3 = filter_300below(filter2, range_data, Instrument)    
    maxBs,filter4 = filter_maxB(filter3, maxB_val) 
    #filter4 = filter_window(filter3, )
       
    loc_beta_f = locate_nan(filter4)
    filtered_S = filter_bynan(loc_beta_f, S)
    return(filtered_S, B_ratio)

###########################################################################
#               ######   Lidar Ratio Filter   ##########
###########################################################################
"""
def step2_Sfilt (S, S_range_percent):
    
    loop through lidar ratio only keeping if within 10% (or other) of
    value before and after
    
    plus = 100. + S_range_percent
    minus = 100. - S_range_percent
    const_S = np.zeros(len(S))
       
    for i in range(len(S)-2):
        if (S[i+1]/S[i]*100 <= plus and S[i+1]/S[i]*100 >= minus and S[i+1]/S[i+2]*100 <= plus and S[i+1]/S[i+2]*100 >= minus):                
            const_S[i+1] = S[i+1]
        else:
            const_S[i+1] = np.nan
    const_S[-1] = np.nan
    const_S[0] = np.nan             ##loop above necessitates neighbouring profiles so first and last index not altered - therefore set to nan   
    return(const_S)
"""

def step2_Sfilt (S, S_range_percent, no_of_profs):
    """
    New version of step2_Sfilt which requires 7 profiles in a row to be
    within 10% of the mean of the 7 profiles. Steps forward one profile 
    at a time. If the profiles meet the criteria, all 7 are copied, else they
    are left as zeros. All remaining zeros converted to nans
    """
    plus = 100. + S_range_percent
    minus = 100. - S_range_percent
    const_S = np.zeros(len(S))
    
    for i in range(len(S)-(no_of_profs - 1)):
        group = S[i:i+(no_of_profs)]
        
        mean_of_group = np.mean(group)
        range_check = group/mean_of_group*100
        if (np.min(range_check) > minus and np.max(range_check) < plus):
            const_S[i:i+(no_of_profs)] = S[i:i+(no_of_profs)]
        else:
            pass
            

    const_S[const_S==0.] = np.nan

    return(const_S)

###########################################################################
#               ######   Cloud Base Height    ##########
###########################################################################
def CBH_check(Step2_S, Instrument,CBH_lower, CBH_max, CBH_min):
    """
    Use CBH as a check for whether to calculate S or not
    Vais currently has an upeer limit to avoid 2.4km issue
    Jen needs a lower limit to avoid saturation at 1.5km, but still need this data for integration so 
    it can't be discarded in above functions..
    Note only CBH_lower used as _middle _upper indicates clouds which do not attenuate (multiple layers)
   
    """

    CBH_lower = np.ma.masked_equal(CBH_lower, -999)
    CBH_accept = np.ma.masked_outside(CBH_lower, CBH_min, CBH_max)
    new_S2 = np.ma.masked_where(np.ma.getmask(CBH_accept), Step2_S)

    return (new_S2)

###############################################################
################# UKV Correction for Water Vapour##############
###############################################################

#ukv_path = '/net/glusterfs/scenario/users/qt013194/Data/ukv/2014/10/20141025_chilbolton_met-office-ukv-0-5.nc'

def WV_Transmission_Profs(ukv_path,range_data, time_data, beta_data):

    obsfile3=Dataset(ukv_path)

    ukv_time = obsfile3.variables['time']
    ukv_height = obsfile3.variables['height']
    ukv_pressure = obsfile3.variables['pressure']
    ukv_temperature = obsfile3.variables['temperature']
    ukv_q = obsfile3.variables['q']

    Rstar = 287 #Gas constant for dry air
    ukv_density_dry = np.array(ukv_pressure)/(Rstar*np.array(ukv_temperature))
    ukv_WV_density = np.array(ukv_q)*ukv_density_dry

    levels = np.array(ukv_height[0,:])/1000 #in km

    #Equations from Ambaum, 2010, pg100
    e = (1.61*np.array(ukv_q)*np.array(ukv_pressure))/(1+(1.61*np.array(ukv_q))-np.array(ukv_q))
    ukv_density2 = (0.0022*e)/ukv_temperature


    levels_m = np.array(ukv_height[0,:])

    ####Interpolate model onto observation space#####
    from scipy.interpolate import griddata

    WV_newgrid = np.zeros((25,(len(range_data.points))))
    for i in xrange(len(WV_newgrid)):
        WV_newgrid[i,:] = griddata(ukv_height[0,:],ukv_density2[i,:] , 1000.0 * range_data.points, method='linear')

    WV_obspace = np.zeros(np.shape(np.transpose(beta_data)))
    for j in xrange(len(np.transpose(WV_obspace))):
        WV_obspace[:,j] = griddata(ukv_time[:],WV_newgrid[:,j] , time_data.points, method='linear')

    ####Calculate Transmissivity####
    #for each WV profile, calculate cumulative integration then calculate transmission at each corresponding height.
    ukv_WV_Beta = np.zeros(np.shape(np.transpose(beta_data)))
    ukv_transmissivity = np.zeros(np.shape(np.transpose(beta_data)))
    ukv_integral = np.zeros(np.shape(np.transpose(beta_data)))
            
    for t in xrange(len(ukv_WV_Beta)):
        ukv_integral[t,:-1] = scipy.integrate.cumtrapz(WV_obspace[t,:],range_data.points[:]*1000)
        ukv_transmissivity[t,:] = 1 - 0.17*(ukv_integral[t,:]*0.1)**0.52
        #set last gate to last but one - not included in integral due to array size matching
        ukv_transmissivity[t,-1] = ukv_transmissivity[t,-2]

    
    return(ukv_transmissivity)




def WV_Transmission_Profs_LUMO(ukv_path1,ukv_path2, range_data, time_data, beta_data):

    obsfile3=Dataset(ukv_path1)
    obsfile4=Dataset(ukv_path2)

    #ukv_time = obsfile3.variables['time']
    ukv_height = obsfile3.variables['height'][:,0,0]
    
    ####PRESSURE####
    #ukv_pressure = []
    ukv_pressure_part1 = obsfile3.variables['P_rhoLev'][0,31:,:,0,0] #Pa
    ukv_pressure_part2 = obsfile4.variables['P_rhoLev'][0,:19,:,0,0] #Pa
    #ukv_pressure = np.concatenate((ukv_pressure, ukv_pressure_part1))
    ukv_pressure = np.concatenate((ukv_pressure_part1, ukv_pressure_part2))

    ####TEMP####
    #ukv_temperature = []
    ukv_temperature_part1 = obsfile3.variables['T_m'][0,31:,:,0,0] #Pa
    ukv_temperature_part2 = obsfile4.variables['T_m'][0,:19,:,0,0] #Pa
    #ukv_temperature = np.concatenate((ukv_temperature, ukv_temperature_part1))
    ukv_temperature = np.concatenate((ukv_temperature_part1, ukv_temperature_part2))

    ####SPECIFIC HUMIDITY####
    #ukv_q = []
    ukv_q_part1 = obsfile3.variables['QV'][0,31:,:,0,0] #Pa
    ukv_q_part2 = obsfile4.variables['QV'][0,:19,:,0,0] #Pa
    #ukv_q = np.concatenate((ukv_q, ukv_q_part1))
    ukv_q = np.concatenate((ukv_q_part1, ukv_q_part2))

    Rstar = 287 #Gas constant for dry air
    ukv_density_dry = np.array(ukv_pressure)/(Rstar*np.array(ukv_temperature))
    ukv_WV_density = np.array(ukv_q)*ukv_density_dry

    levels = np.array(ukv_height[:])#/1000 #in m
    
    ukv_time = [  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,   8.,   9.,  10., 11.,  12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.,  20.,  21., 22.,  23.,  24.]
    
    #Equations from Ambaum, pg100
    e = (1.61*np.array(ukv_q)*np.array(ukv_pressure))/(1+(1.61*np.array(ukv_q))-np.array(ukv_q))
    ukv_density2 = (0.0022*e)/ukv_temperature


    levels_m = np.array(ukv_height[:])

    ####Interpolate model onto observation space#####
    from scipy.interpolate import griddata

    WV_newgrid = np.zeros((25,(len(range_data.points))))
    for i in xrange(len(WV_newgrid)):
        WV_newgrid[i,:] = griddata(ukv_height[:],ukv_density2[i,:] , 1000.0 * range_data.points, method='linear')

    WV_obspace = np.zeros(np.shape(np.transpose(beta_data)))
    for j in xrange(len(np.transpose(WV_obspace))):
        WV_obspace[:,j] = griddata(ukv_time[:],WV_newgrid[:,j] , time_data.points, method='linear')

    ####Calculate Trransmissivity####
    #for each WV profile, calculate cumulative integration then calculate transmission at each corresponding height.
    ukv_WV_Beta = np.zeros(np.shape(np.transpose(beta_data)))
    ukv_transmissivity = np.zeros(np.shape(np.transpose(beta_data)))
    ukv_integral = np.zeros(np.shape(np.transpose(beta_data)))
            
    for t in xrange(len(ukv_WV_Beta)):
        ukv_integral[t,:-1] = scipy.integrate.cumtrapz(WV_obspace[t,:],range_data.points[:]*1000)
        ukv_transmissivity[t,:] = 1 - 0.17*(ukv_integral[t,:]*0.1)**0.52
        #set last gate to last but one - not included in integral due to array size matching
        ukv_transmissivity[t,-1] = ukv_transmissivity[t,-2]

    
    return(ukv_transmissivity)





    
