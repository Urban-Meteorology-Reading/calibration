"""
Code to create block sets of calibration values for ceilometers, including LUMO's.

Created by Emma Hopkin: Mon 11th May 2015
Revised code - bringing together previous work in src file to combine Jen and Vais functions into one and organise layout
Revised 4/1/16 - filteredfor window tau below 90% and requires 7 profiles in a row (this has little if any effect...)

Edited by Elliott Warren: Mon 5th February 2018
"""
#%load_ext autoreload
#%autoreload 2
import sys
# append dir containing EH's utility library
sys.path.append('C:/Users/Elliott/Documents/PhD Reading/LUMO - Sensor network/calibration/utils')

import os # operating system library to issue Unix commands
from os.path import join # use join to link file names to the directory (avoids error with '/' in the path)
import iris  # data management and plotting library from met office
import ncUtils # library for reading netCDF files
import iris.quickplot as qplt # more plotting routines
import matplotlib.pyplot as plt # plotting library (not Met Office)
from netCDF4 import Dataset # to read standard netCDF files
import numpy as np # numerical python library for arrays and mathematical functions
import copy
from scipy import stats
import scipy.io
from scipy.interpolate import interp1d
import LoadData_Utils as LD
import EH_Utils as EH
import Cal_Utils as CAL
#import Cal_Utils_Jen as CAL #############################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!###################################################
import Stats_Utils as ST
#import WV_from_model as MOD
import operator 
import matplotlib
import datetime

#Choose dates to load
#path = '/glusterfs/scenario/users/qt013194/Cubed_Data/MO_Ceils'
#path = '/net/glusterfs/scenario/users/qt013194/Cubed_Data/Elliott'
#mr_path = '/glusterfs/scenario/users/qt013194/Data/microwave_radiometer'
path = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/cubes'

#Location = 'Exeter'
Location = 'LUMO'
Instrument = 'Vais'
CBH_min = 1.0 #for Lufft
CBH_max = 4.0
if Instrument == 'Vais':
    ratio_filt = 0.05
else:
    ratio_filt = 0.10
maxB_filt = -10  #set high so that this filter is no longer used
cont_profs = 5  #number of continuous profiles required for calibration (must be odd no.)
Year =['2016'] # ['2011','2012','2013','2014','2015','2016']
Month = ['01']#['01','02','03','04','05','06','07','08','09', '10', '11', '12']


data_dates = []     #array of string dates yyyy/mm/dd

profile_total = []  #array of total number of profiles used for calibration for each day
peak_total = []     #array of number of profiles in the mode (from histogram)
modes = []          #array of mode of S for each day
means = []          #array of mean of S for each day
medians = []        #array of median of S for each day
sems = []           #array of standard error of S for each day
stdevs = []         #array of standard deviation of S for each day
C_modes = []
C_medians = []
C_stdevs = []
CL_medians = []
CL_stdevs = []


### 2 where relating to aer correction...
modes2 = []          #array of mode of S for each day
means2 = []          #array of mean of S for each day
medians2 = []        #array of median of S for each day
sems2 = []           #array of standard error of S for each day
stdevs2 = []         #array of standard deviation of S for each day
C_modes2 = []
C_medians2 = []
C_stdevs2 = []
CL_medians2 = []
CL_stdevs2 = []
All_S2 = []



modes_wv = []          #array of mode of S for each day
means_wv = []          #array of mean of S for each day
medians_wv = []        #array of median of S for each day
sems_wv = []           #array of standard error of S for each day
stdevs_wv = []         #array of standard deviation of S for each day
C_modes_wv = []
C_medians_wv = []
C_stdevs_wv = []
CL_medians_wv = []
CL_stdevs_wv = []

index_of_maxB = []  #index of max backscatter for each profile (includes all meteorological conditions)
value_of_maxB = []  #value of max backscatter for each profile

Profile_integrations = []       #integral of each profile (0.1-2.4km)

window_tau = []     #instrument reported window transmission for each profile [%]
window_tau_alert=[] #marker if window transmission falls below 90%
pulse_energy = []   #intrument reported pulse energy for each profile [%]
pulse_energy_alert = []#marker if pulse energy falls below 90%
CBH = []            #instrument reported cloud base height [km] - no data => -999
All_S = []          #apparent S for each profile
S_box = []          #apparent S in arrays for box plot of each day

WV_trans = []       #transmission correction by profile from MWR
lengths = []
daily_WV = []       #transmission correction by day from MWR
daymean_modelWV = []#transmission correction by day from model
All_modelWV = []    #transmission correction by profile from model

profiles_in_row = []
file_locs = []

for year in Year:
    #modwv_path = '/data/its-tier2/micromet/data/'+year+'/London/L2/MetOffice/DAY/'
    modwv_path = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/cubes'
    print modwv_path
    for month in Month:
        #list of daily files in the month 
        #filelist = LD.ListToOpen(path,Location,Instrument, year, month)
        filepath = "/".join([path,Location,Instrument, year, month])
        filelist = os.listdir(filepath)
        no_of_files = len(filelist)
        file_locs.append(no_of_files)

        for ll in filelist:
            str_date = year+month+ll[:-3]
            dates = "/".join([year, month, ll[:-3]])
            DOY, DOY2 = EH.get_DOYS(str(dates))
            data_dates.append(dates)
            #For each daily file:
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~' 
            filepath = "/".join([path, Location, Instrument, year, month, ll])
            #filepath = path+'/'+Location+'/'+Instrument+'/'+year+'/'+month+'/'+ll
            print '>>>>',filepath
            daycube = LD.opencube(filepath)
            beta, time_data, range_data = LD.data(daycube)
            if Instrument == 'Jen':
                beta_data = np.transpose(beta.data*10.)###########
                
            else:
                beta_data = np.transpose(beta.data)
                #beta_data = EH.window_correction(beta_data,daycube)     ###window_tau correction
            #quicklook = EH.veiw_quicklook(daycube,beta_data, time_data, range_data)    
            ##Apply scattering correction
            cloud_beta = EH.find_cloud(beta_data)
            if Instrument == 'Jen':
                Scat_correct_b = EH.scatter_correct_Jen(cloud_beta, range_data) 
            else:
                Scat_correct_b = EH.scatter_correct_Vais(cloud_beta, range_data)
                
            beta_arr = EH.corr_beta (Scat_correct_b, beta_data)
            
            #filter out profiles with reduced window transmission
            if Instrument == 'Vais':
                #beta_arr = CAL.filter_window(beta_arr, daycube, 90.)
                #beta_arr = CAL.filter_power(beta_arr, daycube, 90.)
                
                #Revised WV from ukv model
                #ukv_file = str_date+'_chilbolton_met-office-ukv-0-5.nc'
                modwv_file =  'MOUKV_FC'+str_date+'06Z_WXT_KSSW.nc'
                modwv_file2 =  'MOUKV_FC'+str(int(str_date)-1)+'06Z_WXT_KSSW.nc'
                #Chil_ukv_path = '/glusterfs/scenario/users/qt013194/Data/ukv/'

                #def WV_ukv
                #ukv_path = join(Chil_ukv_path,year,month, ukv_file)
                ukv_path2 = "/".join([modwv_path,modwv_file])
                ukv_path1 = "/".join([modwv_path,modwv_file2])

                if(os.path.exists(ukv_path1))  & (os.path.exists(ukv_path2)):
                    ukv_T2_profs = CAL.WV_Transmission_Profs_LUMO(ukv_path1, ukv_path2, range_data, time_data, beta_data)
                    beta_arr_wv = beta_arr*(1/np.transpose(ukv_T2_profs))
                    print 'WV CORRECTION APPLIED'       
                else:
                    beta_arr_wv = beta_arr*np.nan ##set to nan if no water correction available
                    print 'NO WV CORRECTION' 
            #calculate S, including transmission correction
            #S, B_T2 = CAL.lidar_ratio_withT2(beta_arr,range_data, Instrument)
            #
            
            if Instrument == 'Jen':
                beta_arr_saturate = np.copy(beta_arr)

                count_check = 0
                for i in xrange(len(np.transpose(beta_arr_saturate))):
                    index, value = max(enumerate(beta_arr_saturate[:,i]), key=operator.itemgetter(1))
                    prof_from_max = beta_arr_saturate[index:index+33,i] #from max to 500m above
                    neg_check = np.zeros(len(prof_from_max))
                    
                    for j in range(len(prof_from_max)):
                        if prof_from_max[j] < 0:
                            neg_check[j] = neg_check[j-1] + 1.
                        else:
                            neg_check[j] = 0
                    
                    if max(neg_check)>9:    #9=150m, 5 = 100m 
                        count_check = count_check + 1
                        #print max(neg_check),count_check
                        
                        beta_arr_saturate[:,i] = np.nan
                    else:
                        pass

                print 'sat filtered = ', count_check
            
                S,S2 = CAL.lidar_ratio(beta_arr_saturate,range_data, Instrument)
                
            else:
                S,S2 = CAL.lidar_ratio(beta_arr,range_data, Instrument)
                
            ##Apply beta filters            
            Step1_S, profile_B_ratio = CAL.step1_filter(beta_data, range_data,maxB_filt,ratio_filt,S, Instrument)  #aerosol ratio = 5%
            Step1_S2, profile_B_ratio2 = CAL.step1_filter(beta_data, range_data,maxB_filt,ratio_filt,S2, Instrument)  #aerosol ratio = 5%
            ##Apply S Filters
            Step2_S = CAL.step2_Sfilt (Step1_S, 10,cont_profs)            #range in S = 10% 
            Step2_S2 = CAL.step2_Sfilt (Step1_S2, 10,cont_profs) 
            Step2_S[Step2_S <0] = np.nan #(remove neg values caused by neg noise)
            Step2_S2[Step2_S2 <0] = np.nan

            if Instrument == 'Vais':
                S_wv,S_wv2= CAL.lidar_ratio(beta_arr_wv,range_data, Instrument)
                Step1_S_wv, profile_B_ratio_wv = CAL.step1_filter(beta_data, range_data,maxB_filt,ratio_filt,S_wv, Instrument)  #aerosol ratio = 5%
                Step2_S_wv = CAL.step2_Sfilt (Step1_S_wv, 10,cont_profs)  
                Step2_S_wv[Step2_S_wv <0] = np.nan #(remove neg values caused by neg noise)            
                
            ###Count up continuous cal profiles
            it = 0
            for ss in Step2_S:
                if ss<100000:
                    it+=1
                else:
                    profiles_in_row.append(it)
                    it = 0
                if ss == Step2_S[-1]:
                    profiles_in_row.append(it)
                    it = 0
            
            print 'DATE: ',year, month,ll
            

            ##record maxB, CBH                        
            arr = np.copy(beta_data)
            for i in xrange (len(np.transpose(arr))):
                index, value = max(enumerate(arr[:,i]), key=operator.itemgetter(1))
                index_of_maxB.append(index)
                value_of_maxB.append(value)
            
            #record integrated beta from ground to 2.4 km
            int_prof = CAL.total_beta(beta_data, range_data, Instrument)
            Profile_integrations = np.concatenate((Profile_integrations,int_prof))
            
            #CBH_data = daycube.coord('cbh_lower')
            #CBH = np.concatenate((CBH,CBH_data.points))
            
            ##WV at Chilbolton
            if Location == 'MiddleWallop':
                WV_correction = EH.radiometer_file(mr_path,year,str_date,Step2_S)
                day_WV = np.mean(WV_correction)
                WV_trans = np.concatenate((WV_trans,WV_correction))
                daily_WV.append(day_WV)
                lens = len(WV_trans)
                lengths.append(lens)
                
                ####model - create function...
                #ukv_file = str_date+'_chilbolton_met-office-ukv-0-5.nc'
                #daily_modelWV = MOD.WV_model(month, year, ukv_file) #column WV
                #model_corr = EH.WV_Transmission_Linear(daily_modelWV*0.1)#correction
                #day_WV_model = np.mean(model_corr)                          #len 24 - one per hour
                #model_x = np.linspace(0,len(Step2_S),len(daily_modelWV))    #model space
                #new_model_x = np.linspace(0,len(Step2_S),len(Step2_S))      #instrument profile space
                #f=interp1d(model_x,daily_modelWV)
                #modelWV_interpld = f(new_model_x)
                #modelWV_interpld = np.ma.masked_less(modelWV_interpld,0)
                
                #All_modelWV = np.concatenate((All_modelWV, modelWV_interpld))
                
                #daymean_modelWV.append(day_WV_model)

                    
            ##Calculate mode and mean
            if Instrument == 'Jen':
                """
                Impose cloud height min to Lufft
                """
                Step2_S = CAL.CBH_check(Step2_S, Instrument,CBH_data.points, CBH_max, CBH_min)
                Cal_hist, no_of_profs = ST.Plot_ashist(Step2_S.compressed()) #Histogram of filtered S
                no_in_peak, day_mode, day_mean, day_median, day_sem, day_stdev,dayC_mode, dayC_median, dayC_stdev, dayCL_median, dayCL_stdev = ST.S_mode_mean(Step2_S.compressed(), Cal_hist)
            else:
                Cal_hist, no_of_profs = ST.Plot_ashist(Step2_S) #Histogram of filtered S
                no_in_peak, day_mode, day_mean, day_median, day_sem, day_stdev,dayC_mode, dayC_median, dayC_stdev, dayCL_median, dayCL_stdev = ST.S_mode_mean(Step2_S, Cal_hist)
                Cal_hist_wv, no_of_profs_wv = ST.Plot_ashist(Step2_S_wv) #Histogram of filtered S
                no_in_peak_wv, day_mode_wv, day_mean_wv, day_median_wv, day_sem_wv, day_stdev_wv,dayC_mode_wv, dayC_median_wv, dayC_stdev_wv, dayCL_median_wv, dayCL_stdev_wv = ST.S_mode_mean(Step2_S_wv, Cal_hist_wv)


                Cal_hist2, no_of_profs2 = ST.Plot_ashist(Step2_S2) #Histogram of filtered S
                no_in_peak2, day_mode2, day_mean2, day_median2, day_sem2, day_stdev2,dayC_mode2, dayC_median2, dayC_stdev2, dayCL_median2, dayCL_stdev2 = ST.S_mode_mean(Step2_S2, Cal_hist2)




            #plt.close('all')
            All_S = np.concatenate((All_S,Step2_S))
            All_S2 = np.concatenate((All_S2,Step2_S2))
            #~~~~
            S_forbox = np.array(Step2_S)
            S_forbox[np.isnan(S_forbox)] = 0
            S_forbox = S_forbox[np.nonzero(S_forbox)]
            if np.max(Cal_hist)>10:
                S_box.append(S_forbox)
            else:
                S_box.append([0])
                
            profile_total.append(no_of_profs)
            peak_total.append(no_in_peak)
            modes.append(day_mode)
            means.append(day_mean)
            medians.append(day_median)
            sems.append(day_sem)
            stdevs.append(day_stdev)
            C_modes.append(dayC_mode)
            C_medians.append(dayC_median)
            C_stdevs.append(dayC_stdev)            
            CL_medians.append(dayCL_median)
            CL_stdevs.append(dayCL_stdev)
            

            #profile_total2.append(no_of_profs2)
            #peak_total2.append(no_in_peak2)
            modes2.append(day_mode2)
            means2.append(day_mean2)
            medians2.append(day_median2)
            sems2.append(day_sem2)
            stdevs2.append(day_stdev2)
            C_modes2.append(dayC_mode2)
            C_medians2.append(dayC_median2)
            C_stdevs2.append(dayC_stdev2)            
            CL_medians2.append(dayCL_median2)
            CL_stdevs2.append(dayCL_stdev2)


            if Instrument == 'Vais':
                modes_wv.append(day_mode_wv)
                means_wv.append(day_mean_wv)
                medians_wv.append(day_median_wv)
                sems_wv.append(day_sem_wv)
                stdevs_wv.append(day_stdev_wv)
                C_modes_wv.append(dayC_mode_wv)
                C_medians_wv.append(dayC_median_wv)
                C_stdevs_wv.append(dayC_stdev_wv)            
                CL_medians_wv.append(dayCL_median_wv)
                CL_stdevs_wv.append(dayCL_stdev_wv)            
            
            
            ####Additional Queries####
            if Instrument == 'Vais':
                mask_crit = 90 #(%)
                
                window_tau_prof = daycube.coord('window_tau, % unobscured')
                window_monitor = np.ma.masked_greater(window_tau_prof.points, mask_crit)
                window_obscured = len(np.where(window_monitor.mask==False)[0])
                if window_obscured > 0:
                    window_tau_alert.append(1)
                else:
                    window_tau_alert.append(0)
                window_tau = np.concatenate((window_tau,window_tau_prof.points))
                
                pulse_energy_prof = daycube.coord('Laser pulse energy, % of nominal factory setting')
                pulse_monitor = np.ma.masked_greater(pulse_energy_prof.points, mask_crit)
                pulse_drop = len(np.where(pulse_monitor.mask==False)[0])
                if pulse_drop > 0:
                    pulse_energy_alert.append(1)
                else:
                    pulse_energy_alert.append(0)
                pulse_energy = np.concatenate((pulse_energy,pulse_energy_prof.points))         
            else:
                pass
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#Print dates with window_tau or pulse_energy alert
loc_w_alert = np.where(np.asarray(window_tau_alert) == 1)[0]
print '~~~Dates where window_tau is less than threshold~~~'
for w in loc_w_alert:
    print data_dates[w]
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

loc_p_alert = np.where(np.asarray(pulse_energy_alert) == 1)[0]
print '~~~Dates where pulse_energy is less than threshold~~~'
for p in loc_p_alert:
    print data_dates[p]
print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'


#For all files (print to screen):
##Timeseries of daily moes
plt.figure (figsize =(15,6))
plt.title('Daily Lidar Ratio Mode - '+ Location, fontsize = 20)
plt.ylabel ('Apparent S [sr]',fontsize = 16)
plt.yticks(fontsize = 20)
x = np.arange(0,(len(modes)))
print len(x)
print len(modes)
###plot window and pulse alerts
if len(loc_w_alert) > 0:
    for l in loc_w_alert:
        plt.plot(x[l], modes[l], 's', markersize = 16, c = '#F781D8')
        #plt.annotate(data_dates[l], xy = (x[l], modes[l]), xytext = (x[l]-4, modes[l]-1),rotation = 45)
    plt.plot(x[l], modes[l], 's', markersize = 16, label = 'window alert', c = '#F781D8')
if len(loc_p_alert) > 0:   
    for j in loc_p_alert:
        plt.plot(x[j], modes[j], 's', markersize = 12, c = '#A9E2F3')
        #plt.annotate(data_dates[j], xy = (x[j], modes[j]), xytext = (x[j]-5, modes[j]+2),rotation = 325)    
    plt.plot(x[j], modes[j], 'ys', markersize = 12, label = 'pulse alert', c = '#A9E2F3')

###plot daily modes
plt.plot(x,modes, 'x',markeredgewidth = 2, markersize = 10, c = '#0000FF', label = 'Daily Lidar Ratio Mode')
#plt.axis(ymin = 20, ymax = 40, xmin = -1, xmax = len(modes)+1)

m = np.asarray(modes)
m[np.isnan(m)]= 0
m = m[np.nonzero(m)]
mean = np.mean(m)
print 'mean of modes = ', mean
sd = np.std(m)
print 'sd = ', sd
#plt.errorbar(x,modes, yerr = sds)    
plt.plot([-1,len(modes)+1],[mean,mean],'k', label = 'Mean of Daily Modes')
plt.plot([-1,len(modes)+1],[mean+sd,mean+sd], 'k--', label = '1sd ')
plt.plot([-1,len(modes)+1],[mean-sd,mean-sd], 'k--')    


#plt.xticks(a, labels, rotation='vertical', fontsize = 16)
plt.subplots_adjust(bottom=0.17)
plt.legend()
plt.show()
#'''''''''''''''''
print '******************'
mode_sdu, mode_sdu2, mode_sdu3 = mean+sd, mean+sd+sd, mean+sd+sd+sd
mode_sdl, mode_sdl2, mode_sdl3 = mean-sd, mean-sd-sd, mean-sd-sd-sd
N = 'Total N = '+str(len(m))
print N
sd_1 = 'In 1sd = '+str(len(m[(m < mode_sdu) & (m > mode_sdl)]))
print sd_1
sd_2 = 'In 2sd = '+str(len(m[(m < mode_sdu2) & (m > mode_sdl2)]))
print sd_2
sd_3 = 'In 3sd = '+str(len(m[(m < mode_sdu3) & (m > mode_sdl3)]))
print sd_3
#'''''''''''''''''
plt.annotate(N, xy = (0,50))
plt.annotate(sd_1, xy = (0,45))
#plt.annotate(sd_2, xy = (0,66))
#plt.annotate(sd_3, xy = (0,64))
#######################################################################################
##Timeseries of daily means
plt.figure (figsize =(15,6))
plt.title('Daily Lidar Ratio Mean - '+ Location, fontsize = 20)
plt.ylabel ('Apparent S [sr]',fontsize = 16)
plt.yticks(fontsize = 20)
x = np.arange(0,(len(means)))
print len(x)
print len(means)
###Window and pulse alerts
"""
for l in loc_w_alert:
    plt.plot(x[l], means[l], 's', markersize = 16, c = '#F781D8')
plt.plot(x[l], means[l], 's', markersize = 16, label = 'window alert', c = '#F781D8')
for j in loc_p_alert:
    plt.plot(x[j], means[j], 's', markersize = 12, c = '#A9E2F3')    
plt.plot(x[j], means[j], 's', markersize = 12, label = 'pulse alert', c = '#A9E2F3')
"""
###plot daily means
plt.plot(x,means, 'x',markeredgewidth = 2, markersize = 10, c = '#088A08', label = 'Daily Lidar Ratio Mean')
plt.errorbar(x, means, yerr = (np.asarray(sems)),fmt = 'x', markeredgewidth = 2,markersize = 10,label = 'Standard Error')
#plt.axis(ymin = 20, ymax = 100, xmin = -1, xmax = len(modes)+1)

m2 = np.asarray(means)
m2[np.isnan(m2)]= 0
m2 = m2[np.nonzero(m2)]
mean = np.mean(m2)
print 'mean of means = ', mean
sd = np.std(m2)
print 'sd = ', sd
#plt.errorbar(x,modes, yerr = sds)    
plt.plot([-1,len(means)+1],[mean,mean],'k', label = 'Mean of Daily Means')
plt.plot([-1,len(means)+1],[mean+sd,mean+sd], 'k--', label = '1sd ')
plt.plot([-1,len(means)+1],[mean-sd,mean-sd], 'k--')    

#plt.xticks(a, labels, rotation='vertical', fontsize = 16)
plt.subplots_adjust(bottom=0.17)
plt.legend()
plt.show()
#'''''''''''''''''
print '*************************'
mean_sdu, mean_sdu2, mean_sdu3 = mean+sd, mean+sd+sd, mean+sd+sd+sd
mean_sdl, mean_sdl2, mean_sdl3 = mean-sd, mean-sd-sd, mean-sd-sd-sd

N = 'Total N = '+str(len(m2))
print N
sd_1 = 'In 1sd = '+str(len(m2[(m2 < mean_sdu) & (m2 > mean_sdl)]))
print sd_1
sd_2 = 'In 2sd = '+str(len(m2[(m2 < mean_sdu2) & (m2 > mean_sdl2)]))
print sd_2
sd_3 = 'In 3sd = '+str(len(m2[(m2 < mean_sdu3) & (m2 > mean_sdl3)]))
print sd_3
#'''''''''''''''''
plt.annotate(N, xy = (0,48))
plt.annotate(sd_1, xy = (0,46))
plt.annotate(sd_2, xy = (0,44))
plt.annotate(sd_3, xy = (0,55))
#######################################################################################
##Timeseries of daily medians
plt.figure (figsize =(15,6))
plt.title('Daily Lidar Ratio Median - '+ Location, fontsize = 20)
plt.ylabel ('Apparent S [sr]',fontsize = 16)
plt.yticks(fontsize = 20)
x = np.arange(0,(len(medians)))
print len(x)
print len(medians)
###plot window and pulse alerts
"""
for l in loc_w_alert:
    plt.plot(x[l], medians[l], 'rs', markersize = 16, c = '#F781D8')
plt.plot(x[l], medians[l], 'rs', markersize = 16, label = 'window alert', c = '#F781D8')
for j in loc_p_alert:
    plt.plot(x[j], medians[j], 'ys', markersize = 12, c = '#A9E2F3')    
plt.plot(x[j], medians[j], 'ys', markersize = 12, label = 'pulse alert', c = '#A9E2F3')
"""
####plot medians
plt.plot(x,medians, 'x',markeredgewidth = 2, markersize = 12, c = '#DF7401', label = 'Daily Lidar Ratio Median')
plt.axis(ymin = 10, ymax = 80, xmin = -1, xmax = len(modes)+1)

m3 = np.asarray(medians)
m3[np.isnan(m3)]= 0
m3 = m3[np.nonzero(m3)]
mean = np.mean(m3)
print 'mean of medians = ', mean
sd = np.std(m3)
print 'sd = ', sd
#plt.errorbar(x,modes, yerr = sds)    
plt.plot([-1,len(medians)+1],[mean,mean],'k', label = 'Mean of Daily Medians')
plt.plot([-1,len(medians)+1],[mean+sd,mean+sd], 'k--', label = '1sd ')
plt.plot([-1,len(medians)+1],[mean-sd,mean-sd], 'k--')    

#plt.xticks(a, labels, rotation='vertical', fontsize = 16)
plt.subplots_adjust(bottom=0.17)
plt.legend()
plt.show()
#'''''''''''''''''
print '*********************'
median_sdu, median_sdu2, median_sdu3 = mean+sd, mean+sd+sd, mean+sd+sd+sd
median_sdl, median_sdl2, median_sdl3 = mean-sd, mean-sd-sd, mean-sd-sd-sd
N = 'Total N = '+str(len(m3))
print N
sd_1 = 'In 1sd = '+str(len(m3[(m3 < median_sdu) & (m3 > median_sdl)]))
print sd_1
sd_2 = 'In 2sd = '+str(len(m3[(m3 < median_sdu2) & (m3 > median_sdl2)]))
print sd_2
sd_3 = 'In 3sd = '+str(len(m3[(m3 < median_sdu3) & (m3 > median_sdl3)]))
print sd_3
#'''''''''''''''''
#plt.annotate(N, xy = (0,38))
#plt.annotate(sd_1, xy = (0,37))
#plt.annotate(sd_2, xy = (0,36))
#plt.annotate(sd_3, xy = (0,35))
#'''''''''''''''''
#####################################
# As calibration coefficient - using median?? mode better?
plt.figure (figsize =(15,6))
plt.title('Daily Calibration Coefficient - '+ Location+' '+ Instrument, fontsize = 20)
plt.ylabel ('C',fontsize = 16)
plt.yticks(fontsize = 16)
x = np.arange(0,(len(medians)))
med2 = np.array(modes)##########################################################################
if Instrument == 'Jen':
    C_L = (med2/18.8)#*10#*10.e-8
else:
    C_L = med2/18.8
plt.plot(x, C_L,'x',markeredgewidth = 2, markersize = 12, c = 'k', label = 'Calibration Coefficient')
plt.tight_layout()
plt.show()

cc = np.asarray(C_L)
cc[np.isnan(cc)]= 0
cl = cc[np.nonzero(cc)]
cl_mean = np.mean(cl)
print 'cl_mean = ', cl_mean
cl_std = np.std(cl)
print 'cl_std = ', cl_std

#text_file = open("CL31-D.txt", "w")
#text_file.write("Dates: %s" % data_dates)
#text_file.write("C_L: %s" % C_L)
#text_file.close()
"""
text_file = open("ExeterOp_Vais.txt", "w")
text_file.write("CL_day_medians: %s" % C_L)
text_file.write("CL_day_stdevs: %s" % stdevs)
text_file.close()


text_file = open("Calibration_CL31-B_RGS.txt", "w")
text_file.write("Dates: %s" % data_dates)
text_file.write("C_medians: %s" % C_medians)
text_file.write("C_stdevs: %s" % C_stdevs)
text_file.write("C_modes: %s" % C_modes)
text_file.write("profile_total: %s" % profile_total)
text_file.write("peak_total: %s" % peak_total)
text_file.write("C_medians_wv: %s" % C_medians_wv)
text_file.write("C_stdevs_wv: %s" % C_stdevs_wv)
text_file.write("C_modes_wv: %s" % C_modes_wv)
text_file.close()

"""

fig=plt.figure(figsize=(18,4))
plt.ylabel ('Relative Calibration',fontsize = 14)
plt.yticks(fontsize = 12)
x = np.arange(0,(len(medians)))
C_medians = np.asarray(C_medians)
yminus = C_medians -(np.asarray(C_stdevs))
yplus = C_medians +(np.asarray(C_stdevs))
mm = np.isfinite(C_medians)
plt.plot(x[mm], C_medians[mm],'-x',markeredgewidth = 2, markersize = 12, c = 'k', label = 'Calibration Coefficient')
plt.fill_between(x[mm], yminus[mm], yplus[mm], color = '0.7', label = 'Standard Deviation')
ticks = ['Jan11', 'Feb11', 'Mar11', 'Apr11', 'May11', 'Jun11', 'Jul11', 'Aug11','Sept11', 'Oct11', 'Nov11', 'Dec11','Jan12', 'Feb12', 'Mar12', 'Apr12', 'May12', 'Jun12', 'Jul12', 'Aug12','Sept12', 'Oct12', 'Nov12', 'Dec12','Jan13', 'Feb13', 'Mar13', 'Apr13', 'May13', 'Jun13', 'Jul13', 'Aug13','Sept13', 'Oct13', 'Nov13', 'Dec13','Jan14', 'Feb14', 'Mar14', 'Apr14', 'May14', 'Jun14', 'Jul14', 'Aug14', 'Sept14','Oct14', 'Nov14', 'Dec14','Jan15', 'Feb15', 'Mar15', 'Apr15', 'May15', 'Jun15', 'Jul15', 'Aug15', 'Sept15', 'Oct15', 'Nov15', 'Dec15','Jan16', 'Feb16', 'Mar16', 'Apr16', 'May16', 'Jun16']#, 'Jul16','Aug16', 'Sept16', 'Oct16', 'Nov16', 'Dec16']
fff = np.asarray(file_locs)
locs = fff[fff != 0]
locs =  np.cumsum(locs)
locs = np.insert(locs, 0, 0)
plt.xticks(locs, ticks, rotation = 90)
grid()
tight_layout()
plt.show()

fig=plt.figure(figsize=(18,4))
plt.ylabel ('Relative Calibration',fontsize = 14)
plt.yticks(fontsize = 12)
x = np.arange(0,(len(medians)))
C_medians_wv = np.asarray(C_medians_wv)
yminus = C_medians_wv -(np.asarray(C_stdevs_wv))
yplus = C_medians_wv +(np.asarray(C_stdevs_wv))
mm = np.isfinite(C_medians_wv)
plt.plot(x[mm], C_medians_wv[mm],'-x',markeredgewidth = 2, markersize = 12, c = 'k', label = 'Calibration Coefficient')
plt.fill_between(x[mm], yminus[mm], yplus[mm], color = '0.7', label = 'Standard Deviation')
#ticks = ['Jan14', 'Feb14', 'Mar14', 'Apr14', 'May14', 'Jun14', 'Jul14', 'Aug14', 'Sept14','Oct14', 'Nov14', 'Dec14','Jan15', 'Feb15', 'Mar15', 'Apr15', 'May15', 'Jun15', 'Jul15', 'Aug15', 'Sept15', 'Oct15', 'Nov15', 'Dec15','Jan16', 'Feb16', 'Mar16', 'Apr16', 'May16', 'Jun16', 'Jul16','Aug16', 'Sept16', 'Oct16', 'Nov16', 'Dec16']
fff = np.asarray(file_locs)
locs = fff[fff != 0]
locs =  np.cumsum(locs)
locs = np.insert(locs, 0, 0)
plt.xticks(locs, ticks, rotation = 90)
grid()
tight_layout()
plt.show()

"""

####Plot with shaded standard deviation
fig=plt.figure(figsize=(10,4))
plt.title('Daily Calibration Coefficient - '+ Location+' '+ Instrument, fontsize = 20)
plt.ylabel ('C',fontsize = 16)
#plt.ylabel ('CL [x '+ r'$10^{11}$'+']',fontsize = 16)
plt.yticks(fontsize = 16)
x = np.arange(0,(len(medians)))
med2 = np.array(means)
#C_L = med2/18.8
C_L = np.array(C_medians2)
mm = np.isfinite(C_L)
stdevs = np.asarray(C_stdevs2)
yminus = C_L -(np.asarray(stdevs))#/18.8
yplus = C_L +(np.asarray(stdevs))#/18.8
plt.plot(x[mm], C_L[mm],'-x',markeredgewidth = 2, markersize = 12, c = 'r', label = 'Calibration Coefficient')
plt.fill_between(x[mm], yminus[mm], yplus[mm], color = '0.5', label = 'Standard Deviation')
#ticks = ['Jan16', 'Feb16', 'Mar16', 'Apr16', 'May16', 'Jun16', 'Jul16','Aug16', 'Sept16', 'Oct16', 'Nov16']
#ticks = ['Sept14', 'Oct14', 'Nov14', 'Dec14', 'Jan15', 'Feb15', 'Mar15', 'Apr15', 'May15', 'Jun15', 'Jul15', 'Aug15', 'Sept15', 'Oct15', 'Nov15', 'Dec15']
fff = np.asarray(file_locs)
locs = fff[fff != 0]
locs =  np.cumsum(locs)
locs = np.insert(locs, 0, 0)
plt.xticks(locs, ticks, rotation = 90)
grid()
#axis(ymax = 3, ymin = 0)
t = daycube.aux_coords[3].points
plt.title('Daily Calibration Coefficient \n - '+ str(t[0]), fontsize = 18)
plt.legend(loc = 2)
tight_layout()
"""


"""
ticks = ['Jan13', 'Feb13', 'Mar13', 'Apr13', 'May13', 'Jun13', 'Jul13', 'Aug13','Sept13', 'Oct13', 'Nov13', 'Dec13', 'Jan14', 'Feb14', 'Mar14', 'Apr14', 'May14', 'Jun14', 'Jul14', 'Aug14', 'Sept14','Oct14', 'Nov14', 'Dec14']
ticks = ['Sept14', 'Oct14', 'Nov14', 'Dec14', 'Jan15', 'Feb15', 'Mar15', 'Apr15', 'May15', 'Jun15', 'Jul15', 'Aug15', 'Sept15']
locs = [0,31,59,90,120,151,181,212,242,273,304,334,365,396,424, 455,485,516,547,577,608,638,669,699]

ticks = ['Sept14', 'Oct14', 'Nov14', 'Dec14', 'Jan15', 'Feb15', 'Mar15', 'Apr15', 'May15', 'Jun15', 'Jul15', 'Aug15', 'Sept15', 'Oct15', 'Nov15', 'Dec15', 'Jan16', 'Feb16', 'Mar16', 'Apr16', 'May16', 'Jun16', 'Jul16','Aug16', 'Sept16', 'Oct16', 'Nov16', 'Dec16']

locs = [0, 30, 61, 91, 122, 153, 181, 212, 242, 273, 303, 334, 365, 395, 426,456]

locs = [0,31,62,92,123,154,182,213,243,274,304,334,365, 396, 424, 455, ]
plt.xticks(locs, ticks, rotation = 90)
plt.grid()
"""
######################################################################
#CBH Dependecy
# Effect (if any?) on max Beta and B (integrated Beta)
def Saturation_test(CBH, All_S):
    C_lower = np.ma.masked_equal(CBH, -999)
    S = np.ma.masked_array(All_S, np.isnan(All_S))
    plt.figure()
    plt.scatter(S,C_lower, marker = '.')
    plt.xlabel('S [sr]')
    plt.ylabel('CBH_lower [km]')
    return()

def Saturation_test2(CBH, All_S, value_of_maxB):
    M = np.ma.masked_array(maxB,np.isnan(All_S)) #mask profiles not selected for cal
    C_lower = np.ma.masked_equal(CBH, -999)
    plt.figure()
    power = M/((C_lower*1000)**2)
    plt.scatter(power,C_lower)
    plt.axis(xmax = 0.0012, xmin = -0.0002)
    plt.xlabel('Max Beta in Cloud ['+r'$m^{-1}$' + r's$^{-1}$'+']') 
    plt.ylabel('CBH_lower [km]')  
    plt.show()
    return()

ind = np.asarray(index_of_maxB)
MaxB_heights = np.zeros(len(ind))
for i in xrange(len(MaxB_heights)):
    MaxB_heights[i] = range_data.points[ind[i]]
cbh_copy = np.array(CBH)
cbh_copy[cbh_copy == -999.] = np.nan
y = np.ma.masked_array(MaxB_heights, np.isnan(cbh_copy))
x = np.ma.masked_array(value_of_maxB,np.isnan(All_S))
yn = np.ma.masked_array(y, x.mask)
xn = np.ma.masked_array(x, yn.mask)
nx = np.ma.compressed(xn)
ny = np.ma.compressed(yn) 
plt.figure()
plt.hist2d(nx*1000,ny,bins = 50, norm=matplotlib.colors.LogNorm())
#plt.axis(xmax = 0.0012, xmin = -0.0002)
plt.axis(ymax = 4.5, ymin = 0)
plt.xlabel('Max '+r'$\beta$'+' in Cloud ['+r'$km^{-1}$' + r'$sr^{-1}$'+']')
plt.ylabel('Height of Max '+r'$\beta$'+ ' [km]')
#plt.title('Max '+r'$\beta$'+' of filtered S - '+Location+' Vaisala CL31')
plt.title('Max '+r'$\beta$'+' of filtered S - '+Location+' CHM15k Nimbus')
cbar = plt.colorbar()
cbar.set_label('frequency of profiles', rotation = 90)
plt.grid()
plt.show()

"""    
#if Instrument == 'Jen':    
C_lower = np.ma.masked_equal(CBH, -999)    
y = np.ma.masked_equal(CBH, -999)
#x = (np.ma.masked_array(value_of_maxB,np.isnan(All_S)))/(C_lower**2)
###x = np.ma.masked_where(np.ma.getmask(All_S),value_of_maxB )

x = np.ma.masked_array(value_of_maxB,np.isnan(All_S))
yn = np.ma.masked_array(y, x.mask)
xn = np.ma.masked_array(x, yn.mask)
nx = np.ma.compressed(xn)
ny = np.ma.compressed(yn)


plt.figure()
plt.hist2d(nx,ny,bins = 200, norm=matplotlib.colors.LogNorm())
#plt.axis(xmax = 0.0012, xmin = -0.0002)
plt.axis(ymax = 4.5, ymin = 0)
plt.xlabel('Max '+r'$\beta$'+' in Cloud ['+r'$m^{-1}$' + r'$sr^{-1}$'+']')
plt.ylabel('CBH_lower [km]')
plt.title('Max '+r'$\beta$'+' of filtered S - '+Location+' CHM15k Nimbus')
plt.colorbar()
plt.show()
"""
######################################################################
###Test effect of CBH/height of maxB/maxB/Power on C_L
###Uncomment to choosw
height = np.zeros(len(index_of_maxB))
for i in range(len(index_of_maxB)):
    height[i] = range_data.points[index_of_maxB[i]]
"""
C = All_S/18.8
#C[np.isnan(C)]= 0
xx = np.ma.masked_array(C,np.isnan(All_S))
#yy = np.ma.masked_equal(CBH, -999)
yy = np.ma.masked_equal(height, -999)
#yy= np.asarray(value_of_maxB)
#yy= np.asarray(value_of_maxB/(height**2))
yyn = np.ma.masked_array(yy, xx.mask)
xxn = np.ma.masked_array(xx, yyn.mask)
nx = np.ma.compressed(xxn)
ny = np.ma.compressed(yyn)


yy = np.ma.array(height)
xx = np.asarray(value_of_maxB)
xx = np.ma.masked_array(xx,np.isnan(All_S))

biny = np.copy(range_data.points[10:120])
binx = np.arange(0.8,3.5, 0.02)
 binx = np.arange(0.008,0.032,0.0002)

plt.figure()
plt.hist2d(nx,ny,bins = [binx,biny], norm=matplotlib.colors.LogNorm())
plt.yticks(fontsize = 16)
plt.xticks(fontsize = 16)
#plt.axis(ymax = 4, ymin = 0)
plt.xlabel('C')
#plt.ylabel('CBH_lower [km]')
plt.ylabel('Height of max '+r'$\beta$'+' [km]') 
#plt.ylabel('Max '+r'$\beta$'+' in Cloud ['+r'$m^{-1}$' + r'$sr^{-1}$'+']')    
#plt.ylabel('Max Power (max '+r'$\beta$'+' / '+r'$r^{2}$'+')')  
#plt.ylabel('Height of max '+r'$\beta$'+' [km]')
#plt.title('Effect of instrument reported CBH on C - '+Location)
plt.title('Effect of cloud height on C - '+Location)
#plt.title('Max '+r'$\beta$'+' of Calibration Coefficient - '+Location)     
#plt.title('Max Power of Calibration Coefficient - '+Location) 
cbar = plt.colorbar()
cbar.set_label('Number of Profiles', fontsize = 16)
grid()


plt.xlabel('Integrated '+r'$\beta$', fontsize = 16)
plt.title('Integrated '+r'$\beta$'+' of Calibration Profiles \n - '+ str(t[0]), fontsize = 18)


plt.title('Daily Calibration Coefficient \n - '+ str(t[0]), fontsize = 18)
t = daycube.aux_coords[3].points
plt.title('Daily Calibration Coefficient \n - '+ str(t[0]), fontsize = 18)
plt.ylabel('Height of max '+r'$\beta$'+' [km]', fontsize = 16)
plt.xlabel('C', fontsize = 16)
plt.show()
"""

"""

#Calculate mean and std for levels in 2dhist
means_byheight = []
std_byheight = []

start = 0.3
to = start + 0.1
while start <=2.4:
    idx = np.where((ny>=start) & (ny<=to))
    a = [nx[i] for i in idx]
    means_byheight.append(np.mean(a))
    std_byheight.append(np.std(a))
    start+=0.1
    to+=0.1



plt.annotate(str(m[1])+r'$\pm$'+ str(s[1]), xy = (3.0, 0.4))


s = np.round(std_byheight, 2)
m = np.round(means_byheight, 2)
#figure()
#hist = plt.hist2d(nx,ny,bins = [100,100], norm=matplotlib.colors.LogNorm())
#grid()
axis(ymax = 3, ymin = 0)
plt.annotate(str(m[0])+r'$\pm$'+ str(s[0]), xy = (0.03, 0.3))
plt.annotate(str(m[1])+r'$\pm$'+ str(s[1]), xy = (0.03, 0.4))
plt.annotate(str(m[2])+r'$\pm$'+ str(s[2]), xy = (0.03, 0.5))
plt.annotate(str(m[3])+r'$\pm$'+ str(s[3]), xy = (0.03, 0.6))
plt.annotate(str(m[4])+r'$\pm$'+ str(s[4]), xy = (0.03, 0.7))
plt.annotate(str(m[5])+r'$\pm$'+ str(s[5]), xy = (0.03, 0.8))
plt.annotate(str(m[6])+r'$\pm$'+ str(s[6]), xy = (0.03, 0.9))
plt.annotate(str(m[7])+r'$\pm$'+ str(s[7]), xy = (0.03, 1.0))
plt.annotate(str(m[8])+r'$\pm$'+ str(s[8]), xy = (0.03, 1.1))
plt.annotate(str(m[9])+r'$\pm$'+ str(s[9]), xy = (0.03, 1.2))
plt.annotate(str(m[10])+r'$\pm$'+ str(s[10]), xy = (0.03, 1.3))
plt.annotate(str(m[11])+r'$\pm$'+ str(s[11]), xy = (0.03, 1.4))
plt.annotate(str(m[12])+r'$\pm$'+ str(s[12]), xy = (0.03, 1.5))
plt.annotate(str(m[13])+r'$\pm$'+ str(s[13]), xy = (0.03, 1.6))
plt.annotate(str(m[14])+r'$\pm$'+ str(s[14]), xy = (0.03, 1.7))
plt.annotate(str(m[15])+r'$\pm$'+ str(s[15]), xy = (0.03, 1.8))
plt.annotate(str(m[16])+r'$\pm$'+ str(s[16]), xy = (0.03, 1.9))
plt.annotate(str(m[17])+r'$\pm$'+ str(s[17]), xy = (0.03, 2.0))
plt.annotate(str(m[18])+r'$\pm$'+ str(s[18]), xy = (0.03, 2.1))
plt.annotate(str(m[19])+r'$\pm$'+ str(s[19]), xy = (0.03, 2.2))
plt.annotate(str(m[20])+r'$\pm$'+ str(s[20]), xy = (0.03, 2.3))
plt.annotate(str(m[21])+r'$\pm$'+ str(s[21]), xy = (0.03, 2.4))
axis(xmax = 4)
axis(xmax = 3.5)
plt.xlabel('C', fontsize = 16)
plt.ylabel('Height of max '+r'$\beta$'+' [km]', fontsize = 16)
plt.yticks(fontsize = 16)
"""

###
#For power v height
""" 
xx = np.asarray(value_of_maxB/(height**2))
xx = np.ma.masked_array(xx,np.isnan(All_S))
yy = np.ma.masked_greater(height, 4.)
"""
#then use as above 
 
###################################################################### 
###
#Water Vapour Correction
plt.figure(figsize =(15,6))
plt.plot((All_S*WV_trans)/All_S)
plt.ylabel('(S_App*WV_Correction)/S_App')
aa=lengths[::15]
bb=data_dates[::15]
plt.xticks(aa,bb, rotation = 90)
plt.title('WV Correction at Middle Wallop \n ratio of WV Corrected S_App to Uncorrected S_App')
plt.tight_layout()
plt.grid()
plt.axis(ymin =0.5, ymax = 1)
plt.show()

modelWV_trans = EH.WV_Transmission_Linear(All_modelWV*0.1)

"""
fff = np.asarray(file_locs)
locs = fff[fff != 0]
locs =  np.cumsum(locs)
locs = np.insert(locs, 0, 0)
locs = locs*(4148)
fig=plt.figure(figsize=(18,4))

plot(window_tau)
axis(ymax = 110, ymin = 0)
#plt.xticks(locs, ticks, rotation = 90)
grid()
title(t[0] + ' (CL31-B)', fontsize = 14)
ylabel('Window Transmission [%]', fontsize = 14)
tight_layout()
plt.show()

fig=plt.figure(figsize=(18,4))
plot(pulse_energy)
axis(ymax = 110, ymin = 0)
#plt.xticks(locs, ticks, rotation = 90)
grid()
title(t[0] + ' (CL31-B)', fontsize = 14)
ylabel('Pulse energy [%]', fontsize = 14)
tight_layout()
"""
#CL31-A
#ticks = ['Mar2011', 'Apr2011', 'Feb2012', 'Mar2014', 'Oct2014', 'Jan2015', 'Mar2015', 'Aug2015', 'Oct2015', 'Dec2015', 'Jan2016', 'Mar2016'] 
#CL31-D
#ticks = ['Mar2011', 'Nov2011', 'Feb2012', 'Nov2012', 'Dec2012', 'Jan2013', 'Mar2014', 'Oct2014', 'Apr2015', 'Nov2015','Mar2016', 'May2016'] 
#CL31-D
#ticks = ['Feb2012', 'Jan2013', 'Nov2013','Mar2014', 'Oct2014', 'Apr2015', 'Jul2015','Nov2015','Mar2016'] 
#CL31-D
#ticks = ['Mar2011', 'Nov2011', 'Feb2012','Jan2015', 'Feb2015','Mar2015', 'Dec2015', 'Mar2016', 'May2016'] 

#for i in range(len(means)):                            
#       print i, means[i]

##Timeseries of window transmission
##Timeseries of laser pulse energy
##Timeseries of CBH - report if more than 1 layer?


################################   WATER VAPOUR PLOTS   ########################

fig=plt.figure(figsize=(10,8))

ax1 = fig.add_subplot(211)
ax1.plot(x, C_medians,'x',markeredgewidth = 2, markersize = 10, c = 'k', label = 'C (without WV correction)')
ax1.plot(x, C_medians_wv,'x',markeredgewidth = 2, markersize = 10, c = 'r', label = 'C (with WV correction)')
ax1.legend(fontsize = 10)
#ticks = ['Jan14', 'Feb14', 'Mar14', 'Apr14', 'May14', 'Jun14', 'Jul14', 'Aug14', 'Sept14','Oct14', 'Nov14', 'Dec14','Jan15', 'Feb15', 'Mar15', 'Apr15', 'May15', 'Jun15', 'Jul15', 'Aug15', 'Sept15', 'Oct15', 'Nov15', 'Dec15','Jan16', 'Feb16', 'Mar16', 'Apr16', 'May16', 'Jun16', 'Jul16','Aug16', 'Sept16', 'Oct16', 'Nov16', 'Dec16']
fff = np.asarray(file_locs)
locs = fff[fff != 0]
locs =  np.cumsum(locs)
locs = np.insert(locs, 0, 0)
plt.xticks(locs, ticks, rotation = 90)
plt.yticks(fontsize = 14)
plt.ylabel('C', fontsize = 14)
plt.title('Effect of Water Vapour on Calibration Coefficient\n - '+ str(t[0]), fontsize = 16)
ax1.grid()


ax2 = fig.add_subplot(212) 
ax2.plot(x, np.asarray(C_medians) - np.asarray(C_medians_wv), 'x',markeredgewidth = 2, markersize = 10, c = 'k')
plt.xticks(locs, ticks, rotation = 90)
plt.yticks(fontsize = 14)
ax2.grid()
plt.ylabel('Difference in C', fontsize = 14)
plt.title('Size of Water Vapour Impact on Calibration Coefficient\n - '+ str(t[0]), fontsize = 16) 
plt.tight_layout()
plt.show()



















