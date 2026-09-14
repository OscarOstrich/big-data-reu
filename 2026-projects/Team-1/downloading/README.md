# Downloading

In this folder, the files are used to download all data for reproducing our study for No2 air quality forecasting with Aurora. The pipeline was designed to run the files individually for downloading into a particular directory which must be added into the files. 

1. [Download CAMS](download_cams.py)
2. [Download TEMPO](download_tempo_cloud_by_day.py)
3. Download Aurora Predictions *this depends on how many days 1, 2, or 3

*Aurora cannot generate predictions without the correct library being installed, and we used the Air Pollution model version with CAMS input variables, meaning that it cannot be run without the CAMS being downloaded first
