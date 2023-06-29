# AdaBoost_wear

# AdaBoost_wear is a Python software that implements the AdaBoost algorithm to predict the coefficient of friction (COF) of B83 babbitt alloy as a function of time.

# Installation

# To install AdaBoost_wear, you need to have Python 3.6 or higher and the following packages:

# numpy
# pandas
# sklearn
# matplotlib
# seaborn
# You can install these packages using pip or conda.

# To download AdaBoost_wear, you can clone this repository using git:

git clone https://github.com/mihail-15/AdaBoost_wear.git

# Alternatively, you can download the zip file from https://github.com/mihail-15/AdaBoost_wear/archive/refs/heads/main.zip and extract it to your desired location.

# Usage

# To use AdaBoost_wear, you need to have the experimental data of time and friction force from pin-on-disk tests for each material and load combination in XLSX files. The files should be stored in the data folder and have the following names: B83_40N.xlsx, B83_50N.xlsx, and B83_60N.xlsx.

# To run AdaBoost_wear, you can execute the main.py script from the command line:

python main.py

# The script will read the data files, train and test the AdaBoost model, calculate the performance metrics, and generate the plots. The results will be displayed on the console and stored in the results folder.

# License

# AdaBoost_wear is licensed under the MIT License. See LICENSE.txt for more details.
