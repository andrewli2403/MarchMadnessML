# MarchMadnessML
March Madness Machine Learning is using 13 years worth of march madness datasets to predict the annual collegiate basketball tournament.

Simulator.py allows for the projections of the 2020 March Madness Tournament.

MarchMadness2020.py contains all 14 specific datapoints for each of the 64 collegiate teams in the 2020 Tournament.

CollegeBasketballData.ipynb is a data processing file that converts Raw Data CSV files into appropriate 2D PyTorch arrays.

main.py is where the neurnal network is located. Comprised of of linear and non-linear layers.

HyperParamterTuning.py is the testing which optimal learning rate is the most accurate for the given model. Saved model with best learning rate is stored in optimal_learningmodel.pt

UtilityFunctions.py is where loading PyTorch datasets, building the model, and other functions are stored.

