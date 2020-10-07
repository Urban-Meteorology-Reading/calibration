library(ncdf4)
library(sys)
library(dplyr)

#get args 
cla <- commandArgs(trailingOnly = T)
progDir <- cla[1]
source(file.path(progDir, 'utils', 'applyCalibrationToBSCFunctions.R'))
baseDir <- cla[2]
sIds <- cla[3]
yrs <- cla[4]

#format years and site ids
siteIds <- strsplit(sIds, ';')[[1]]
years <- strsplit(yrs, ';')[[1]]

#for every site
for (siteId in siteIds){
  print(siteId)
  
  # get site and inst info
  site <- strsplit(siteId, '_')[[1]][2]
  instId <- strsplit(siteId, '_')[[1]][1]
  # for every year
  for (year in years){
    print(year)
    #read in calibration file
    c_data <- getC_data(baseDir, year, site, instId)
    if (is.null(c_data)) next
    
    #get year start and end date
    startDate <- as.Date(paste0(c(year, '1', '1'), collapse = '-'))
    endDate <- as.Date(paste0(c(year, '12', '31'), collapse = '-'))
    proDate <- seq.Date(startDate, endDate, 'day')
    #loop through dates apply calibration to L0 BSC
    for (i in 1:length(proDate)){
      DATE <- proDate[i]
      print(DATE)
      #get day of year
      doy <- strftime(DATE, '%j')
      
      #extract cal value for day
      dayCal <- c_data %>% dplyr::filter(c_DATE == DATE)
      #if there is no cal value skip
      if(nrow(dayCal) == 0){
        message('No calibration value for this day')
        next
      } else if (is.na(dayCal[['c_pro']])){
        message('No calibration value for this day')
        next
      } 
      
      #read in L1 BSC
      bscFileCon <- getBSCFileCon(baseDir, year, doy, site, instId)
      if (is.null(bscFileCon)) next
      
      #multiply backscatter by cal value
      dayBSCCal <- applyCal(bscFileCon, dayCal)
      
      # get copy of BSC ncdf var 
      bscCalVar <- bscFileCon$var
      
      #create new file
      bscCalFileCon <- getBSCFileCon(baseDir, year, doy, site, instId, 
                                     calibrate = T, vars = bscCalVar)
      # add new calibrated BSC
      ncvar_put(bscCalFileCon, 'BSC', as.matrix(dayBSCCal))
      #copy global attrbutes from old to new file
      addGlobalAtts(bscFileCon, bscCalFileCon)
      
      #close files
      nc_close(bscCalFileCon)
      nc_close(bscFileCon)
    }
  }
}

print('Program finished')
