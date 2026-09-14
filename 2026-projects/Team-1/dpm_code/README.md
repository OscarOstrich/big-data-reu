# Direct Prediction Model

The direct prediction model code for each of our 3 different types of models. We decided to use Normalizing Flows, Extreme Gradient Boosting, and a Random Forest approach for comparison. Although each model has different code structure, they all accomplish the same end goal which is to directly predict the TEMPO ground truth of our experiment via the CAMS input data. 

Random Forest utilizes the pycache included within the folder, while the others can be run stand alone as long as their RESIDUAL CORRECTION VERSIONS have been run first. The pathing and horizon lead times would have to be changed to run, but otherwise they should be able to run with their respective dependencies. 

This is the last folder to be run after downloading all data, preprocessing by creating and splitting the dataset, and running the residual correction. 
