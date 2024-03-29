# MarchMadnessML
March Madness Machine Learning is using 13 years worth of march madness datasets to predict the annual collegiate basketball tournament. Every March, bracket projections are posted after NCAA's Selection Sunday.

# Main Files
`Simulator.py` allows for the projections of the 2020 March Madness Tournament.

`MarchMadness2020.py` contains all 14 specific datapoints for each of the 64 collegiate teams in the 2020 Tournament.

`CollegeBasketballData.ipynb` is a data processing file that converts [Raw Data CSV](https://www.kaggle.com/c/march-machine-learning-mania-2017/data) files into appropriate 2D PyTorch arrays.

`main.py` is where the neural network is located. Comprised of linear and non-linear layers.

`HyperParamterTuning.py` is the testing which optimal learning rate is the most accurate for the given model. Saved model with best learning rate is stored in `optimal_learningmodel.pt`

`UtilityFunctions.py` is where loading PyTorch datasets, building the model, and other functions are stored.

# 2021 Projection
<img src="images/2021 MarchMadnessResults.png" width = 800>

# 2020 Projection
<img src="images/2020 MarchMadnessResults.png" width = 800>
