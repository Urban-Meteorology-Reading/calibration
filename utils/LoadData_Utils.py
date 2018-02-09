"""
Emma Hopkin
11th May 2015
Functions to open daily cubes
either as a single cube or as a monthly list
"""

import os # operating system library to issue Unix commands
from os.path import join # use join to link file names to the directory (avoids error with '/' in the path)
from os import path, system
#import ceilometer_utils as cutils, cristina's utilities. you won't need this yet.
import iris  # data management and plotting library from met office
import ncUtils # library for reading netCDF files
import iris.quickplot as qplt # more plotting routines
import matplotlib.pyplot as plt # plotting library (not Met Office)
from netCDF4 import Dataset # to read standard netCDF files
import numpy as np # numerical python library for arrays and mathematical functions
import copy
########################################################################
#List of files to open
def ListToOpen(path,Location, Instrument, Year, Month):
    """
    Create list of files to open
    Input: Location, Instrument, Year, Month
    eg ('Eskdalemuir', 'Jen' , '2014', '09')
    Output: Array of filenames
    """

    #path to list of files
    #path = '/home/qt013194/Project_Calibration/data/'
    filepath = join(str(path), str(Location), str(Instrument), str(Year), str(Month))
    #create file with list of files
    #list_of_saved_files = "ls_out"
    #command = os.listdir(filepath)
    #command = "ls "+filepath+" > "+join(filepath,list_of_saved_files)
    #print 'CHECK ListToOpen: ', command
    print '..................................'
    os.system(command)
    print type(filepath), filepath
    if os.path.isdir(filepath):
        #loop through to import into array
        fn = open(join(filepath,list_of_saved_files))
        lines = [line.strip() for line in fn]
        lines.remove(list_of_saved_files);
        fn.close()
    else:
        print 'ERROR: Directory does not exist'
        lines = []   
    return (lines)
    
#----------------------------------------------------------------#
########################################################################
#SIRTA List of files to open
def SIRTAListToOpen(Location, Instrument, Year, Month):
    """
    Create list of files to open
    Input: Location, Instrument, Year, Month
    eg ('Eskdalemuir', 'Jen' , '2014', '09')
    Output: Array of filenames
    """

    #path to list of files
    path = '/net/glusterfs/scenario/users/qt013194/Cubed_Data/'
    filepath = join(path, str(Location), str(Year), str(Month))
    #create file with list of files
    list_of_saved_files = "ls_out"
    command = "ls "+filepath+" > "+join(filepath,list_of_saved_files)
    print 'CHECK ListToOpen: ', command
    os.system(command)
    if os.path.isdir(filepath):
        #loop through to import into array
        fn = open(join(filepath,list_of_saved_files))
        lines = [line.strip() for line in fn]
        lines.remove(list_of_saved_files);
        fn.close()
    else:
        print 'ERROR: Directory does not exist'
        lines = []
    return (lines)
    
#----------------------------------------------------------------##functions to open a single cube file by user input
def which_cube():
    """
    Select date of cube to be opened
    """
    year_to_open = str(raw_input ('year of file (format YYYY): '))
    month_to_open = str(raw_input ('month of file (format MM): '))
    day_to_open = str(raw_input ('day of file (format DD): '))
    file_to_open = join(year_to_open, month_to_open,day_to_open)
    file_date = str(file_to_open) + '.nc' 
    print file_date
    return (file_date)

def opencube(filename):
    """
    Open file
    """
    daycube_aslist = iris.load(filename)
    daycube = daycube_aslist[0]
    return (daycube)
    


###Call above 2 functions all together    
def return_cube(file_loc):
    """
    Runs functions to select and open cube 
    - reduce functions called in each program
    Returns the cube
    """
    ceil_loc = raw_input('Ceilometer Location: ')
    file_date = which_cube()
    #file_loc = '/home/qt013194/Project_Calibration/data' 
    filename = join(file_loc, ceil_loc, file_date)
    daycube = opencube(filename)
    return (daycube)  
    
      
def data(daycube):
    #beta, time, range
    beta_data = np.ma.masked_less(daycube.data, 1e-12)
    time_data = daycube.dim_coords[0]
    range_data = daycube.dim_coords[1]
    return (beta_data, time_data, range_data)    
#----------------------------------------------------------------# 
def cube_toload (file_loc, ceil_loc, Instrument, file_date):
    """
    as return_cube function above but removes raw_input to function input
    """
    ceil_loc = join(ceil_loc, Instrument)
    filename = join(file_loc, ceil_loc, file_date)
    daycube = opencube(filename)
    return (daycube)  
























   

