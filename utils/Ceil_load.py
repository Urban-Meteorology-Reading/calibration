"""
15/05/15
Emma Hopkin

Load in data from simone (Martial Haeffelin)
"""
import os # operating system library to issue Unix commands
from os.path import join # use join to liNK file names to the directory (avoids error with '/' in the path)
#import ceilometer_utils as cutils, cristina's utilities. you won't need this yet.
import iris  # data management and plotting library from met office
import ncUtils # library for reading netCDF files
import iris.quickplot as qplt # more plotting routines
import matplotlib.pyplot as plt # plotting library (not Met Office)
from netCDF4 import Dataset # to read standard netCDF files
import numpy as np # numerical python library for arrays and mathematical functions
import copy
from os import system
import sys
import datetime
print '###############################################################'
geog_cs = iris.coord_systems.GeogCS(semi_major_axis=6378137, inverse_flattening=298.257223563)

YYYY = str('2016')
#datapath = '/data/its-tier2/micromet/data/'+YYYY+'/London/L1/NK/DAY'
#datapath = '/net/glusterfs/micromet/users/micromet/Emma_Calibration/L1/CL31-D/'+YYYY+'/'
datapath = '/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/L1/'

#fileday = str(sys.argv[1][-3:])
fileday = '019'
print '>>', fileday
#filename = 'CL31_BSC_NK_2015'+fileday+'_15sec.nc'
#filename = 'CL31-D_BSC_NK_'+YYYY+fileday+'_15sec.nc'
filename = 'CL31-D_BSC_NK_2016019_15sec.nc'
filetoload = join(datapath, filename)
#filetoload = join(datapath, filename)
print filename
print type(filename)
print 'FILE TO LOAD: ', filetoload

##############################################################################################
#datapath_diags = '/net/glusterfs/micromet/users/micromet/Emma_Calibration/L0/CL31-D/'+YYYY+'/'
datapath_diags = '/data/its-tier2/micromet/data/'+YYYY+'/London/L0/NK/DAY/'+fileday+'/'
filename_diags = 'CL31-D_CLD_NK_'+YYYY+fileday+'_15sec.nc'

datapath_diags = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/L0/'
filename_diags = 'CL31-D_CLD_NK_2016019_15sec.nc'

filetoload_diags = join(datapath_diags, filename_diags)
print 'DIAGNOSTIC FILE TO LOAD: ', filetoload_diags
list_of_contents_diags=ncUtils.nc_listatt(filetoload_diags)
print 'DIAGNOSTIC  Attributes : ', list_of_contents_diags
list_variables_diags=ncUtils.nc_listvar(filetoload_diags)
print 'DIAGNOSTIC Variables: ',list_variables_diags
obsfile_diags=Dataset(filetoload_diags)
##############################################################################################




list_of_contents=ncUtils.nc_listatt(filetoload)
print ' Attributes : ', list_of_contents



#Assign meta data in the netcdf file to variables
system_name =  ncUtils.nc_getatt(filetoload, 'Institution')

#dday = str(ncUtils.nc_getatt(filetoload, 'day'))
#if len(dday) == 1:
#    dday = '0'+dday
#else:
#    pass
#mmonth = str(ncUtils.nc_getatt(filetoload, 'month'))
#if len(mmonth) == 1:
#    mmonth = '0'+mmonth
#else:
#    pass
the_year = int(filename[-16:-12])
the_day = int(filename[-12:-9])
the_date = datetime.datetime(the_year, 1, 1) + datetime.timedelta(the_day - 1)
date = the_date.strftime('%d/%m/%Y')

dday = date[:2] 
mmonth = date[3:5]
yyear = date[6:]

#yyear = str(ncUtils.nc_getatt(filetoload, 'year'))
ddate =  str(dday)+'-'+str(mmonth)+'-'+str(yyear)
file_title =  ncUtils.nc_getatt(filetoload, 'Title')
location=  ncUtils.nc_getatt(filetoload, 'Site')
#lat = ncUtils.nc_getatt(filetoload, 'lat')
#lon = ncUtils.nc_getatt(filetoload, 'long')
       
list_variables=ncUtils.nc_listvar(filetoload)
print 'Variables: ',list_variables 

obsfile=Dataset(filetoload)
time = obsfile.variables['time']
time_points = obsfile.variables['time'][:]/3600 #(time in hours)

lat = obsfile.variables['lat'][:]
lon = obsfile.variables['lon'][:]

range = obsfile.variables['height']
range_points = obsfile.variables['height'][:]/1000

#CBH = obsfile_diags.variables['CBH']
#CBH1 = obsfile_diags.variables['CLD_Height_L1'][:,0,0,0]
#CBH2 = obsfile_diags.variables['CLD_Height_L2'][:,0,0,0]
#CBH3 = obsfile_diags.variables['CLD_Height_L3'][:,0,0,0]
#CBH = []
#CBH.append(CBH1)
#CBH.append(CBH2)
#CBH.append(CBH3)
#CBH = np.asarray(CBH)
#CBHunit='m'

pulse_energy = obsfile_diags.variables['pulse'][:,0,0,0]

window_tau = obsfile_diags.variables['transmission'][:,0,0,0]

#att_power = obsfile.variables['BSC']
#att_power_points = obsfile.variables['BSC'][:]      
#Multiply by calibration constant(10^-8) to get attenuated backscatter
#beta =  att_power_points*(1e-8)     
#beta = beta[:,:,0,0]

beta = obsfile.variables['beta'][:,:,0,0]
beta = beta*(1e-12)        
# Create Iris cube called "simone_cube" to hold the netcdf data with coordinate system (backscatter and all meta-data)
#  http://scitools.org.uk/iris/docs/latest/index.html        
        
time_units = 'Time in hours since '+ddate+' 00:00:00'
print iris.util.monotonic(time_points, strict=True)
print type(time_points)
b_time_coord=iris.coords.DimCoord(time_points,long_name=time_units, var_name='time', units = None)       

b_range_coord=iris.coords.DimCoord(range_points, long_name = 'Range [km]', var_name='range', units=range.units)
                                    
simone_cube=iris.cube.Cube(beta,long_name='Apparent attenuated backscatter',units='m-1 sr-1',dim_coords_and_dims=[(b_time_coord, 0), (b_range_coord, 1)])        
        
New_coord = iris.coords.AuxCoord(pulse_energy[:],long_name = 'Laser pulse energy, % of nominal factory setting', units = '%')
simone_cube.add_aux_coord(New_coord, (0,))

New_coord2 = iris.coords.AuxCoord(window_tau[:],long_name = 'window_tau, % unobscured', units = '%')
simone_cube.add_aux_coord(New_coord2, (0,))  

#for i, name in enumerate(['lower', 'middle', 'upper']):
#   coord = iris.coords.AuxCoord(CBH[i,:], units=CBHunit,
#                             long_name='cbh_{}'.format(name))
#simone_cube.add_aux_coord(coord, (0,))

simone_cube.add_aux_coord(iris.coords.AuxCoord((str(lat)[0:6]),
                                                        standard_name='latitude', 
                                                        coord_system=geog_cs)
                                   )
simone_cube.add_aux_coord(iris.coords.AuxCoord((str(lon)[0:6]),
                                                        standard_name='longitude', 
                                                        coord_system=geog_cs)
                                        )                       
                                        
simone_cube.add_aux_coord(iris.coords.AuxCoord(str(file_title)))

                                                       
simone_cube.add_aux_coord(iris.coords.AuxCoord(str(location),
                                                        long_name='site_name')
                                        )
 #def save_cube(simone_cube, the_site, the_date):
"""
input: simone_cube
output: cube of required variable for one day saved
"""

the_date = str(yyear)+'/'+str(mmonth)+'/'+str(dday)
#filename = '/home/qt013194/Project_Calibration/data/Martial/'
#filename = '/net/glusterfs_essc/scenario/users/qt013194/Cubed_Data/Simone/NK/LUMO'
#filename = '/net/glusterfs/scenario/users/qt013194/Cubed_Data/Elliott/CL31-D/Vais'
filename = 'C:/Users/Elliott/Documents/PhD Reading/PhD Research/Aerosol Backscatter/clearFO/data/cubes/'
the_date = the_date + '.nc'

file_tosave = join(filename,  'cube20160119.nc')
print file_tosave
#directory = file_tosave[:-5]
#system("mkdir -p "+directory) #need path minus file name...
#print 'SAVEPATH: ',file_tosave
iris.fileformats.netcdf.save(simone_cube, file_tosave, netcdf_format='NETCDF4')



############################################################################

def data(simone_cube):
    #beta, time, range
    beta_data = np.ma.masked_less(simone_cube.data, 1e-12)
    time_data = simone_cube.dim_coords[0]
    range_data = simone_cube.dim_coords[1]
    return (beta_data, time_data, range_data)


beta, time_data, range_data = data(simone_cube)

#PLot Attenuated Backscatter Quicklook - temp - delete section...
def quicklook(simone_cube,beta_data, time_data, range_data):
    """
    Input: 24hr cube
    Output: quicklook plot
    """
    # Set up standard levels for the logarithmic plotting of backscatter
    std_beta_levels=np.linspace(-7,-3, 8)
    beta_levels_option1=np.linspace(-9.0,-3.2, 10)
    
    fig=plt.figure(figsize=(14,4))
    temp = (np.ma.log10(beta_data))
    TITLE = simone_cube.aux_coords[2].points
    #cf=plt.pcolormesh(time_data.points,range_data.points , temp)
    cf=plt.contourf(time_data.points,range_data.points , temp,levels=std_beta_levels)
    plt.xlabel('Time in '+str(time_data.units), fontsize = 18)
    plt.ylabel('Range ['+str(range_data.units)+']', fontsize = 18)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.title(simone_cube.long_name+': '+ str(TITLE[0]), fontsize = 18)
    cbar = plt.colorbar(cf, orientation='vertical', fraction = 0.1)
    cbar.set_label('Backscatter 10^ [m-1 sr-1]', fontsize = 18)
    #file2save = 'Attenuated_Backscatter_'+str(date)+'.png'
    #plt.savefig(join(webdir,file2save),dpi=100, bbox_inches='tight')
    #plt.axis(ymax = 2.5)
    plt.tight_layout()
    plt.show()   
    filetosave = 'quicklook' +'.png' 
    return(filetosave)
     
beta, time_data, range_data = data(simone_cube)        
beta_data = np.transpose(beta.data)
#beta_quicklook = quicklook(simone_cube, beta_data, time_data, range_data)        
        
        
        

        
        
        
        
        
        
        
