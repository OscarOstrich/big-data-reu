# Residual Correction Model

The residual correction model code for each of our 3 different types of models. We decided to use Normalizing Flows, Extreme Gradient Boosting, and a Random Forest approach for comparison. Although each model has different code structure, they all accomplish the same end goal which is to predict the TEMPO - Aurora residual No2 number per pixel on a given date/ time step. 

Random Forest utilizes the pycache included within the folder. The pathing and horizon lead times would have to be changed to run, but otherwise they should be able to run with their respective dependencies.

This is the second to last folder to be run after downloading all data, and preprocessing by creating and splitting the dataset.
