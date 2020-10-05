### supplementary functions for applyCalirationToBSC.R ###

getC_data <- function(baseDir, year, site, instId){
  ## read annual calibration file ##
  
  #configure file directory
  L2CalDir <- file.path(baseDir, 'data', year, 'London', 'L2', site, 'ANNUAL') 
  L2CalFileName <- paste0(paste0(c(instId, 'CAL', site, year), collapse = '_'),'.nc')
  #open file
  if (file.exists(file.path(L2CalDir, L2CalFileName))){
    #open CAL file
    calFileCon <- nc_open(file.path(L2CalDir, L2CalFileName))
    #get calibration info
    c_pro <- ncvar_get(calFileCon, 'c_pro')
    c_days <- calFileCon$dim$time$vals
    c_firstDate <- as.Date(strsplit(calFileCon$dim$time$units, ' ')[[1]][3])
    c_dates <- c_firstDate + c_days
    c_data <- data.frame(c_DATE = c_dates, c_pro = c_pro)
    #close cal connection
    nc_close(calFileCon)
    
    return(c_data)
    
  } else{
    message('No calibration file for this instrument for this year')
    return(NULL)
  }
}

getBSCFileCon <- function(baseDir, year, doy, site, instId, calibrate = F, vars = NULL){
  ## open file connection with BSC file ##
  
  #get paths
  bscDir <- file.path(baseDir, 'data', year, 'London', 'L0', site, 'DAY', doy)
  
  if (calibrate == T){
    bscFile <- paste0(paste0(c(instId, 'BSC', 'calibrated', site, 
                               paste0(year,doy), '15sec'), 
                           collapse = '_'),'.nc')
    #create file
    bscFileCon <- nc_create(file.path(bscDir,bscFile), vars = vars)
    
  } else {
    bscFile <- paste0(paste0(c(instId, 'BSC', site, 
                               paste0(year,doy), '15sec'), 
                             collapse = '_'),'.nc')
    #open file
    if (file.exists(file.path(bscDir, bscFile))){
      bscFileCon <- nc_open(file.path(bscDir, bscFile))
    } else{
      message('No BSC file for tis day')
      bscFileCon <- NULL
    }
  }
  return(bscFileCon)
}

applyCal <- function(bscFileCon, dayCal){
  ## multiply backscatter by cal value ##
  
  #get backscatter values
  dayBSC <- ncvar_get(bscFileCon, 'BSC')
  
  #multiply backscatter by cal 
  dayBSCCal <- data.frame(dayBSC) %>% 
    mutate_all(function(x){x*dayCal[['c_pro']]})
  
  return(dayBSCCal)
}

addGlobalAtts <- function(bscFileCon, bscCalFileCon){
  ## copy global attributes from olf file to new ##
  globalAtts <- ncatt_get(bscFileCon, 0)
  globalAttNames <- names(globalAtts)
  for (att in globalAttNames){
    if (att == 'Title'){
      Title <- 'CL31_BSC_calibrated'
      ncatt_put(bscCalFileCon, 0, att, Title)
    } else if (att == 'QAQC'){
      QAQC <- 'Scaled by calibration value calculated with https://github.com/Urban-Meteorology-Reading/ceilometer_calibration'
      ncatt_put(bscCalFileCon, 0, att, QAQC)
    } else {
      ncatt_put(bscCalFileCon, 0, att, globalAtts[[att]])
    }
  }
}