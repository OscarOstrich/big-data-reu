# Preprocessing 

This folder is for keeping preprocessing files that organize and transform the Aurora forecasts, TEMPO scans, and CAMS inputs into an easily useable data table CSV. As mentioned in the other README.md files, the primary part to change for running is just the file paths for your own directories, otherwise they should be able to run as they are with their respective dependencies. 

The order of files to be run is as follows:

1. Organize Aurora and TEMPO
   - move the forecasts and truths into their own directories for downstream analysis
  
2. [Generate the Dataset](/preprocessing/gen_timeseries.py)
   - create a time series data set that takes two time steps to use for horizon lead times
  
3. [Split the Train and Test](/preprocessing/split_timeseries_dataset.py)
   - splits the generated data set into training and testing based on inputted dates


*The split time series can be customized to fit different amounts of time depending on the inputted dates, as long as the files have been moved using the organization scripts to have the same naming convention. 
