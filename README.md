# ceilometer_calibration
calibration code for ceilometers originally developed by Dr. Elliot Warren, expanded on from Emma Hopkin's work.

See [this wiki entry](https://github.com/Urban-Meteorology-Reading/Observation_Equipment_Docs/wiki/Ceilometer-calibration) for more info on the reason for and theory of this repo.

## Usage 

### ceilometer_periods

These files are used to remove calibration values from days in which an event that will have altered the window transmission occurred. This includes firmware changes, hardware changes, window cleaning etc. The information is stored in csv files with the headers 'Date', 'Reason' and 'Type'. The 'Reason' and 'Type' columns are not particularly important as all dates in the list are filtered out! The ceilometer_periods are stored [here](https://github.com/Urban-Meteorology-Reading/ceilometer_calibration/tree/master/ceilometer_periods)

### Interpolation type 

In L2 the calibration value are interpolated by "window_transmission", "block_avg" or "time". You define this [here](https://github.com/Urban-Meteorology-Reading/ceilometer_calibration/blob/c7f5a81fb16c804ceed834e47fe10062b1234a0d/scripts/create_calibration_L2.py#L58-L92) in the code. Check [this section](https://github.com/Urban-Meteorology-Reading/Observation_Equipment_Docs/wiki/Ceilometer-calibration#chosing-interpolation-type) of the wiki page for advice on interpolation type.

### Args for submission

The python script takes 4 arguments:
  
  1. Directory where the programme is stored 
  
  2. Base directory (relative to data/YEAR/London/LEVEL/SITE/DAY/DOY) to ceilometer data. At the time of writing the base directory is $MM_DAILYDATA or /storage/basic/micromet/Tier_raw/ on the RACC.
  
  3. Years to process. If multiple these should be separated by ';'. E.g. to run for 2018, 2019 and 2020 this arg will be 2018;2019;2020
  
  4. Instruments to process. This is in the format CL31-LETTER_SITE e.g. CL31-D_SWT. If multiple these should be separated by ';'. E.g. CL31-D_SWT;CL31-A_IMU;CL31-E_HOP.
  
 ## Process 
 
 ### Creating L0 calibrated backscatter
 
 Three scripts are required to be run in order:
 
   1. **create_calibration_L1.py**: Makes daily calibration values whenever there was cloud and saves them in yearly files. **Output**: base_dir/data/YEAR/London/L1/SITE/ANNUAL/CL31-LETTER_CAL_SITE_YEAR.nc
   
   2. **create_calibration_L2.py**: Interpolates the calibration values onto days when there wasn't any cloud (like clear sky days). **Output**: base_dir/data/YEAR/London/L2/SITE/ANNUAL/CL31-LETTER_CAL_SITE_YEAR.nc
   
   3. **applyCalibrationToBSC.R**: multiplies the backscatter by the calibration value for that day. **Output**: base_dir/data/YEAR/London/L0/SITE/DAY/DOY/CL31-LETTER_BSC_calibrated_SITE_YEARDOY_15sec.nc 
 
#### Running on the RACC

At the time of writing, there exists a script on the RACC to call these processes: `$HOME/script/slurmScripts/ceilometer-calibration.sh`. Example uasage: `cd $HOME/script/slurmScripts/submitScript; ./MMsbatch.sh ceilometer_calibration.sh --export=PROG_VERSION=2808871,years="2019;2020",site_ids="CL31-D_SWT;CL31-A_IMU",base_dir=$MM_DAILYDATA --time=800 --mem=10G --mail-user=k.j.benjamin@reading.ac.uk --mail-type=FAIL`

### Getting calibrated L1 BSC and MLH 
Once the L0 calibrated backscatter is created, the L1 BSC and MLH files can be created using calibrated L0 backscatter. This can be done using [this branch](https://github.com/Urban-Meteorology-Reading/Operations-CEIL/tree/calibration) of ceilometer processing with an *calibrate* being an environment variable, set to TRUE. E.g. submission on the RACC: `cd $HOME/script/slurmScripts/submitScript; ./MMsbatch.sh CEIL.sh --export=PROG_VERSION=BRANCH_HASH,DATE=$(date -d "-1 day" +"%j%Y"),calibrate=TRUE --time=60 --mem=10G --mail-user=$MM_FIXING_EMAIL --mail-type=FAIL`
