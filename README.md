# ceilometer_calibration
calibration code for ceilometers originally developed by Dr. Elliot Warren, expanded on from Emma Hopkin's work.

See [this wiki entry](https://github.com/Urban-Meteorology-Reading/Observation_Equipment_Docs/wiki/Ceilometer-calibration) for more info on the reason for and theory of this repo.

## Usage 

### ceilometer_periods

These files are used to remove calibration values from days in which an event that will have altered the window transmission occurred. This includes firmware changes, hardware changes, window cleaning etc. The information is stored in csv files with the headers 'Date', 'Reason' and 'Type'. The 'Reason' and 'Type' columns are not particularly important as all dates in the list are filtered out!
