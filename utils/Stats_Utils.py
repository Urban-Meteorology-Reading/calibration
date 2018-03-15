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

######################################################################################
#plot as histograms
import numpy as np
def Plot_ashist(variable):
    v = np.copy(variable)
    v[np.isnan(v)] = 0
    vartoplot = v[np.nonzero(v)]
    if len(vartoplot) > 2:
        b = (np.round(np.max(vartoplot)))
        
        #plt.figure(figsize = (6,4))
        counts, bins, range = plt.hist(vartoplot, bins = (2*b), range = (0,b))
        #plt.xticks(np.arange(0, 110, 10.0), fontsize = 18)
        plt.yticks(fontsize = 18)
        #plt.axis(xmax = b+1, xmin = 16)
        bin_centers = 0.5 * np.diff(bins) + bins[:-1]
        
        plt.title('S',fontsize = 20)
        plt.xlabel('Apparent S [sr]', fontsize = 18)
        plt.ylabel('Frequency', fontsize = 18)
        plt.tight_layout()
        
        #plt.axis(xmax = 100)
        #plt.show()
    else:
        counts = 0    
    return(counts,len(vartoplot)) 



def S_mode_mean (Step2_S, Cal_hist):
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
        
        m[np.isnan(m)]= 0
        m = m[np.nonzero(m)]
        mean = np.mean(m)           ###mean   
        print 'calibration mean = ', mean
        
        stdev = np.std(m)
        print 'std deviation = ', stdev        
        
        sem = stats.sem(m)
        print 'std error = ', sem
        
        m2 = np.round((m.tolist()),1)
        mode_arr = stats.mode(m2)
        mode = mode_arr[0][0]
        print 'calibration mode = ', mode
        
        median = np.median(m)
        print 'calibration median = ', median
        
        C_data = m/18.8
        
        C_median =np.median(C_data)
        print 'C median = ', C_median        
        
        C_mode_arr = stats.mode(m2/18.8)
        C_mode = C_mode_arr[0][0]
        print 'C mode = ', C_mode
        
        C_stdev = np.std(C_data)
        print 'C std = ', C_stdev
        
        CL_data = 1./(m/18.8)
        
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
    return (peak,mode, mean, median, sem, stdev,C_mode, C_median, C_stdev, CL_median, CL_stdev)
    
    
###########################################################################
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return (ret[n - 1:] / n)    
    
   
