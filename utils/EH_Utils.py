"""
Emma Hopkin
11th May 2015
Functions!
"""
import os # operating system library to issue Unix commands
from os.path import join # use join to link file names to the directory (avoids error with '/' in the path)
#import ceilometer_utils as cutils, cristina's utilities. you won't need this yet.
import iris  # data management and plotting library from met office
#import ncUtils # library for reading netCDF files
import iris.quickplot as qplt # more plotting routines
import matplotlib.pyplot as plt # plotting library (not Met Office)
from netCDF4 import Dataset # to read standard netCDF files
import numpy as np # numerical python library for arrays and mathematical functions
import copy
import operator
from scipy.interpolate import interp1d
import datetime
#############################################################################

def find_cloud(beta_data):
    """
    Find the location of the max beta
    Take cloud range as 15 gates above and below
    return array with 30gate range beta only; others set to nan
    """
    arr = np.copy(beta_data)
    for i in xrange (len(np.transpose(arr))):
       

        index, value = max(enumerate(arr[:,i]), key=operator.itemgetter(1))
        loc_above = index + 15 
        loc_below = index - 15
        if loc_below < 0:
            loc_below = 0
        else:
            pass
        arr[loc_above:,i] = np.nan #set everything above to nan
        arr[:loc_below,i] = np.nan #set everything below to nan
    return(arr)
    

###Scattering correction is instrument dependent - Vais or Lufft options

def scatter_correct_Vais(cloud_beta, range_data):
    """
    Input: raw beta data
    Output: beta with scattering correxction applied
    Apply multiple scattering correction - source:  http://www.met.reading.ac.uk/~swr99ejo/lidar_calibration/index.html
    """
    Scat_correct_b = np.copy(cloud_beta)
    #range_km = range_data.points
    range_km = range_data.points
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


def scatter_correct_Jen(cloud_beta, range_data):
    """
    Input: raw beta data
    Output: beta with scattering correxction applied
    Apply multiple scattering correction - source:  http://www.met.reading.ac.uk/~swr99ejo/lidar_calibration/index.html
    """
    Scat_correct_b = np.copy(cloud_beta)
    
    ind1 = np.where(range_data.points< 0.250)
    Scat_correct_b[ind1,:]=Scat_correct_b[ind1,:]* 0.83063
    ind2 = np.where((np.abs(range_data.points - 0.375)) < 0.125)
    Scat_correct_b[ind2,:]=Scat_correct_b[ind2,:]*0.82969
    ind3 = np.where((np.abs(range_data.points - 0.625)) < 0.125)
    Scat_correct_b[ind3,:]=Scat_correct_b[ind3,:]*0.82806
    ind4 = np.where((np.abs(range_data.points - 0.875)) < 0.125)
    Scat_correct_b[ind4,:]=Scat_correct_b[ind4,:]*0.82614
    ind5 = np.where((np.abs(range_data.points - 1.125)) < 0.125)
    Scat_correct_b[ind5,:]=Scat_correct_b[ind5,:]*0.82382	
    ind6 = np.where((np.abs(range_data.points - 1.375)) < 0.125)
    Scat_correct_b[ind6,:]=Scat_correct_b[ind6,:]*0.82079
    ind7 = np.where((np.abs(range_data.points - 1.625)) < 0.125)
    Scat_correct_b[ind7,:]=Scat_correct_b[ind7,:]*0.81780
    ind8 = np.where((np.abs(range_data.points - 1.875)) < 0.125)
    Scat_correct_b[ind8,:]=Scat_correct_b[ind8,:]*0.81457
    ind9 = np.where((np.abs(range_data.points - 2.125)) < 0.125)
    Scat_correct_b[ind9,:]=Scat_correct_b[ind9,:]*0.81072
    ind10 = np.where((np.abs(range_data.points - 2.375)) < 0.125)
    Scat_correct_b[ind10,:]=Scat_correct_b[ind10,:]*0.80716
    ind11 = np.where((np.abs(range_data.points - 2.625)) < 0.125)
    Scat_correct_b[ind11,:]=Scat_correct_b[ind11,:]*0.80353           
    ind12 = np.where((np.abs(range_data.points - 2.875)) < 0.125)
    Scat_correct_b[ind12,:]=Scat_correct_b[ind12,:]*0.79940    
    ind13 = np.where((np.abs(range_data.points - 3.125)) < 0.125)
    Scat_correct_b[ind13,:]=Scat_correct_b[ind13,:]*0.79573
    ind14 = np.where((np.abs(range_data.points - 3.375)) < 0.125)
    Scat_correct_b[ind14,:]=Scat_correct_b[ind14,:]*0.79209
    ind15 = np.where((np.abs(range_data.points - 3.625)) < 0.125)
    Scat_correct_b[ind15,:]=Scat_correct_b[ind15,:]*0.78807
    ind16 = np.where((np.abs(range_data.points - 3.875)) < 0.125)
    Scat_correct_b[ind16,:]=Scat_correct_b[ind16,:]*0.78457        
    return (Scat_correct_b)

#MERGE SCATTERING CORRECTION WITH BETA DATA
def corr_beta (Scat_beta, beta_data):
    """
    Locate nans placed by finding cloud(see above)
    replace with beta values
    s_cor_beta is array of beta with scatting correction applied to 30 gates around max beta
    """
    s_cor_beta = np.copy(Scat_beta)

    for prof in  xrange (len(np.transpose(s_cor_beta))):
        index_n = np.isnan(Scat_beta[:,prof])
        thenans = np.where(index_n == True)
        for locnan in xrange(len(thenans)):
           s_cor_beta[(thenans[locnan]), prof] = beta_data [(thenans[locnan]), prof]
       
       
    return(s_cor_beta)  
    
    
    
#################################################################
###QUICKLOOK###
def veiw_quicklook(cube,beta_data, time_data, range_data, Instrument):
    """
    Input: 24hr cube
    Output: quicklook plot
    """
    # Set up standard levels for the logarithmic plotting of backscatter
    std_beta_levels=np.linspace(-7,-3, 8)
    beta_levels_option1=np.linspace(-9.0,-3.2, 10)
    
    fig=plt.figure(figsize=(12,4))
    temp = (np.ma.log10(beta_data))
    if Instrument == 'Vais':
        TITLE = cube.aux_coords[3].points
    elif Instrument == 'Jen':
        TITLE = cube.aux_coords[3].points
    else:
        TITLE = [' ']
    cf=plt.pcolormesh(time_data.points,range_data.points , temp,vmin = -7, vmax = -3)
    #cf=plt.contourf(time_data.points,range_data.points , temp,levels=std_beta_levels)
    plt.xlabel('Time in '+str(time_data.units), fontsize = 18)
    plt.ylabel('Range ['+str(range_data.units)+']', fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.title(cube.long_name+': '+ str(TITLE[0]), fontsize = 18)
    cbar = plt.colorbar(cf, orientation='vertical', fraction = 0.1)
    cbar.set_label('Backscatter 10^ [m-1 sr-1]', fontsize = 18)
    #file2save = 'Attenuated_Backscatter_'+str(date)+'.png'
    #plt.savefig(join(webdir,file2save),dpi=100, bbox_inches='tight')
    plt.axis(xmax = 24, ymax =  np.max(range_data.points))
    plt.tight_layout()
    plt.show()   
    filetosave = 'quicklook' +'.png' 
    return(filetosave)  

def veiw_quicklook2(cube,beta_data, time_data, range_data, Instrument):
    """
    Input: 24hr cube
    Output: quicklook plot
    """
    # Set up standard levels for the logarithmic plotting of backscatter
    std_beta_levels=np.linspace(-7,-3, 8)
    beta_levels_option1=np.linspace(-9.0,-3.2, 10)
    
    fig=plt.figure(figsize=(12,4))
    temp = (np.ma.log10(beta_data))
    if Instrument == 'Vais':
        TITLE = cube.aux_coords[5].points
    elif Instrument == 'Jen':
        TITLE = cube.aux_coords[3].points
    else:
        TITLE = [' ']
    cf=plt.pcolormesh(time_data.points,range_data.points , temp,vmin = -7, vmax = -3)
    #cf=plt.contourf(time_data.points,range_data.points , temp,levels=std_beta_levels)
    plt.xlabel('Time in '+str(time_data.units), fontsize = 18)
    plt.ylabel('Range ['+str(range_data.units)+']', fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.title(cube.long_name+': '+ str(TITLE[0]), fontsize = 18)
    cbar = plt.colorbar(cf, orientation='horizontal', fraction = 0.1)
    cbar.set_label('Backscatter 10^ [m-1 sr-1]', fontsize = 18)
    #file2save = 'Attenuated_Backscatter_'+str(date)+'.png'
    #plt.savefig(join(webdir,file2save),dpi=100, bbox_inches='tight')
    plt.axis(xmax = 24, ymax =  np.max(range_data.points))
    plt.tight_layout()
    plt.show()   
    filetosave = 'quicklook' +'.png' 
    return(filetosave)
    
####S TIMESERIES###
#plot timeseries of filtered data overlaid with constant S
def plot_Stimeseries(all_S,Step1_S,Step2_S,time_data):
    plt.figure(figsize =(11,4))
    all_S = np.log10(all_S)
    #plt.plot(time_data.points,all_S, c = '#CECEF6', label = 'S')
    plt.plot(time_data.points,Step1_S, c = '#E2A9F3', label = 'Step 1 Filtering Only',linewidth = 2, solid_capstyle="butt")
    plt.plot(time_data.points, Step2_S, label = 'Step 1 and 2 Filtering', linewidth = 1, solid_capstyle="butt")
    plt.title('Single Profile Lidar Ratios', fontsize = 20)
    plt.xlabel(time_data.units, fontsize = 18)
    plt.ylabel('Apparent S [sr]', fontsize = 18)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    #plt.axis (ymax = 400)
    plt.legend() 
    plt.tight_layout()
    plt.show()
    filetosave = 'Const_S' +'.png'
    return(filetosave)    
    
    
#########################################################################
###Window_tau correction for vais###
def window_correction(beta_data,daycube):
    window_tau_prof = daycube.coord('window_tau, % unobscured')
    beta_data_new = np.zeros(np.shape(beta_data))
    for j in range(len(window_tau_prof.points)) :        
            beta_data_new[:,j] = (beta_data[:,j])/(window_tau_prof.points[j]/100)
    return(beta_data_new)
    
#########################################################################
########            WV Transmission Loss Estimate           #############
#########################################################################
def WV_Transmission():
    x = np.array([0,0.5, 1.0, 2.0, 5.0])
    y = np.array([1.0,0.89, 0.84, 0.75, 0.61])
    #z = np.polyfit(x,y,2)
    xp = np.linspace(0,6,100)
    p = np.poly1d(z)
    return (p)
    #WV_correction = (p[0]*__**2) + (p[1]*__) + (p[2])
    
def WV_Transmission_Linear(WV):
    WV_correction =  1 - 0.17*WV**0.52
    return (WV_correction)
        
##############Get Radiometer data#########################    
def radiometer_file(mr_path,year,dates,Step2_S):
    mr_file = dates+'_chilbolton-radiometer-radiometrics.nc'
    mr_filepath = join(mr_path,year,mr_file)
    print 'WV FILE: ', mr_filepath
    if os.path.exists(mr_filepath):
        obsfile=Dataset(mr_filepath)
        atm_wv_content = obsfile.variables['atmosphere_water_vapor_content']
        precip_water = atm_wv_content[:]*0.1 #units g/cm2
        x=np.linspace(0,len(Step2_S),len(precip_water))
        xnew=np.linspace(0,len(Step2_S),len(Step2_S))
        f=interp1d(x,precip_water)
        int_precip = f(xnew)
        
        #p = WV_Transmission()
        #WV_correction = p(int_precip)
        WV_correction =  WV_Transmission_Linear(int_precip)
    else:
        print 'FILE DOES NOT EXIST'
        print 'No WV correction'
        WV_correction = np.zeros(len(Step2_S))
    return(WV_correction)
    
    
def WV_prof_correction(beta_prof,WV_correction,Cloud_max_loc):
    Prof_WV = np.copy(beta_prof)
    if np.isnan(Cloud_max_loc) == True:
        Prof_WV[:] = np.nan
    else:
        correction_factors = np.zeros(len(beta_prof))
        correction_factors[:Cloud_max_loc] = np.linspace(1, (1/WV_correction), Cloud_max_loc)
        correction_factors[Cloud_max_loc:] = np.linspace((1/WV_correction),(1/WV_correction),(len(beta_prof) -Cloud_max_loc))
        Prof_WV[0:Cloud_max_loc]= beta_prof[0:Cloud_max_loc]*correction_factors[:Cloud_max_loc]
        Prof_WV[Cloud_max_loc:] = beta_prof[Cloud_max_loc:]*correction_factors[Cloud_max_loc:]
    return (Prof_WV)
    
    
#################################################################################    

def get_DOYS(str_date):
    """
    Get doy of year and previous day of year for selecting two water vapour files needed to make up the 24hrs
    """
    fmt = '%Y/%m/%d'
    dt = datetime.datetime.strptime(str_date,fmt)
    tt = dt.timetuple()

    a = tt.tm_yday
    b = a-1
    a = str(a)
    b = str(b)
    #Create DOY in format ###
    if len(a) == 2:    
	    DOY1 = '0'+str(a)
    elif len(a) ==1:
	    DOY1 = '00'+str(a)
    elif len(a) == 3:
	    DOY1 = a
    else:
	    print 'Error in date format'
    #Create previous day in format ###
    if len(b) == 2:    
	    DOY2 = '0'+str(b)
    elif len(b) ==1:
        DOY2 = '00'+str(b)
    elif len(b) ==3:
        DOY2 = b	    
    else:
	    print 'Error in date format'

    return(DOY1, DOY2)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
             
