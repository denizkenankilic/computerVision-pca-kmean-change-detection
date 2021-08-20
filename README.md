# computerVision-pca-kmean-change-detection
PCA &amp; K-Mean Based Change Detection Algorithm with Performance Calculator

Reference for the algorithm:

Celik, T. Unsupervised change detection in satellite images using principal component analysis and k-means clustering. IEEE Geoscience and Remote Sensing Letters, 2009, 6(4), 772-776. https://doi.org/10.1109/LGRS.2009.2025059

The "cd2014_thermal_read_arguments.bat" is used to trigger "PCAKmeans.py" code to calculate and write the change maps. 4th, 6th and 7th arguments in each line in the batch file need to be edited according to file names in "CD_Codes/Celik/cd2014_thermal". The "calculate_performances.bat" is used to trigger "changeDetectionPerformanceTool.py" for calculating all performances with respect to ground truth images. The desired error metrics are specified in the "Inputs_of_Change_Detection_Performance_Tool.xls" files. The ones that are wanted to be calculated should be written as TRUE and the ones that are not wanted to be estimated as FALSE.
