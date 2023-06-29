# import the necessary packages
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor # import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib import patheffects

# Code lines for material B83 (B)

# Read each .xlsx file containing the data for each dataset of the material B83 with 40 N load

file_B1 = pd.read_excel("B83_40N_1.xlsx") 
file_B2 = pd.read_excel("B83_40N_2.xlsx") 
file_B3 = pd.read_excel("B83_40N_3.xlsx") 

# Assign the dataset to a variable 

time_B1 = file_B1["TIME"] # time in sec. for B1
friction_B1 = file_B1["FF"] # friction force in N  for B1
time_B2 = file_B2["TIME"] # time in sec. for B2
friction_B2 = file_B2["FF"] # friction force in N  for B2
time_B3 = file_B3["TIME"] # time in sec. for B3
friction_B3 = file_B3["FF"] # friction force in N  for B3

# Calculation of the COF for each dataset using the formula: COF = friction force / normal  force (40 N)


cof_B1 = friction_B1 / 40 # COF for  B1 
cof_B2 = friction_B2 / 40 # COF for  B2 
cof_B3 = friction_B3 / 40 # COF for  B3 



# Create an empty list to store the average COF 

avg_cof_B_list= [] 


# Create an empty list to store the average time 

avg_time_B_list= []  

# Average COF 
for i in range(0, len(time_B1)):
    avg_cof_B_list.append((cof_B1[i] + cof_B2[i] + cof_B3[i]) / 3)


# Average time
    avg_time_B_list.append((time_B1[i] + time_B2[i] + time_B3[i]) / 3)


# For material B
mean_cof_B = np.mean(avg_cof_B_list) # calculate the mean
median_cof_B = np.median(avg_cof_B_list) # calculate the median
std_cof_B = np.std(avg_cof_B_list) # calculate the standard deviation
min_cof_B = np.min(avg_cof_B_list) # calculate the minimum
max_cof_B = np.max(avg_cof_B_list) # calculate the maximum

# Print the descriptive statistics for B
print("Descriptive statistics for material B:")
print("Mean: {:.4f}".format(mean_cof_B))
print("Median: {:.4f}".format(median_cof_B))
print("Standard deviation: {:.4f}".format(std_cof_B))
print("Minimum: {:.4f}".format(min_cof_B))
print("Maximum: {:.4f}".format(max_cof_B))

# Save the descriptive statistics to a file
with open('B_descriptive_statistics.txt','w') as f:
 f.write('Descriptive statistics for material B:\n')
 f.write('Mean: {:.4f}\n'.format(mean_cof_B))
 f.write('Median: {:.4f}\n'.format(median_cof_B))
 f.write('Standard deviation: {:.4f}\n'.format(std_cof_B))
 f.write('Minimum: {:.4f}\n'.format(min_cof_B))
 f.write('Maximum: {:.4f}\n'.format(max_cof_B))
 f.close()


# Save the list as a new excel file using pd.DataFrame() and pd.to_excel()

avg_cof_B_df= pd.DataFrame(avg_cof_B_list, columns=["Average Coefficient Of Friction B"]) 
avg_cof_B_df.to_excel("Average_COF_B.xlsx", index=False) 

avg_time_B_df= pd.DataFrame(avg_time_B_list, columns=["Average Time B"]) 
avg_time_B_df.to_excel("Average_time_B.xlsx", index=False) 

# read the data from the two excel files
avg_cof_B_df = pd.read_excel("Average_COF_B.xlsx") 
avg_time_B_df = pd.read_excel("Average_time_B.xlsx")

# concatenate the data from the two files into one DataFrame
merged_df = pd.concat([avg_time_B_df, avg_cof_B_df], axis=1)

# save the new DataFrame to an Excel file
merged_df.to_excel("COF_B.xlsx", index=False)


# load the data from the 'xlsx' file (in our case "pred_COF_B.xlsx")
data = pd.read_excel(r"COF_B.xlsx")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# define a parameter grid with the hyperparameters and their ranges
param_grid = {
   'n_estimators': [50, 100, 200], # number of weak learners
   'learning_rate': [0.1, 0.5, 1.0], # weight of each weak learner
   'loss': ['exponential'], # loss function
   'base_estimator': [DecisionTreeRegressor(max_depth=3)] # base estimator
}

# create an AdaBoost regressor
abr = AdaBoostRegressor(random_state=42) # set random state for reproducibility

# create a grid search object with 5-fold cross-validation
gs = GridSearchCV(abr, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

# fit the grid search on the training data

gs.fit(X_train, y_train)


# print the best parameters and score
print("Best parameters: ", gs.best_params_)
print("Best score: ", gs.best_score_)

# get the best estimator
dt_best = gs.best_estimator_

# predict the coefficient of friction for the test set

y_pred = dt_best.predict(X_test)

# predict the coefficient of friction for the validation set
y_val_pred = dt_best.predict(X_val)

# calculate the R2 score for both sets
r2_test = r2_score(y_test,y_pred)
r2_val = r2_score(y_val,y_val_pred)

# calculate and store the performance metrics for both sets in a file (for "B_performance_metrics.txt")
with open('B_performance_metrics.txt','w') as f:
 f.write('Test set performance metrics:\n')
 f.write('R2 score: {:.4f}\n'.format(r2_test))
 f.write('RMSE: {:.4f}\n'.format(np.sqrt(mean_squared_error(y_test,y_pred))))
 f.write('MSE: {:.4f}\n'.format(mean_squared_error(y_test,y_pred)))
 f.write('MAE: {:.4f}\n'.format(mean_absolute_error(y_test,y_pred)))
 f.write('Validation set performance metrics:\n')
 f.write('R2 score: {:.4f}\n'.format(r2_val))
 f.write('RMSE: {:.4f}\n'.format(np.sqrt(mean_squared_error(y_val,y_val_pred))))
 f.write('MSE: {:.4f}\n'.format(mean_squared_error(y_val,y_val_pred)))
 f.write('MAE: {:.4f}\n'.format(mean_absolute_error(y_val,y_val_pred)))
 f.close()


# Create figure 1
fig1 = plt.figure()



# Shadow effect objects with different transparency and smaller linewidth
pe1 = [patheffects.SimpleLineShadow(offset=(0.5,-0.5), alpha=0.4), patheffects.Normal()]

# Plot of the actual vs predicted COF as a function of time
plt.scatter(X_test[:, 0], y_test,color='cyan',label='Actual test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.scatter(X_test[:, 0], y_pred,color='orange',label='Predicted test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.scatter(X_val[:, 0], y_val,color='green',label='Actual val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.scatter(X_val[:, 0], y_val_pred,color='magenta',label='Predicted val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe1)
plt.xlabel('Time, s', fontsize='15', fontweight='bold')
plt.ylabel('Coefficient of friction, -', fontsize='15', fontweight='bold')
plt.legend(shadow=True, prop={'size':'12'}, loc='lower right')

# x axis limit 
plt.xlim(0 ,450)

# y axis limit 
plt.ylim(0 ,0.3)

# gridlines to the plot
plt.grid(True)


# Add a title
plt.title("B83 40 N", fontsize='18', fontweight='bold')

plt.show()

# Save the plot with dpi=500 in 'png'
fig1.savefig('pred_COF_B.png', dpi=500)


# Close plot material B
plt.close(fig1)

# create a DataFrame from the variables
df = pd.DataFrame({"Actual test": y_test, "Predicted test": y_pred, "Actual val": y_val, "Predicted val": y_val_pred})
# save the DataFrame to an Excel file
df.to_excel("test_val_data_B.xlsx", index=False)


# For material B
fig3 = plt.figure() # assign a variable to the figure object
plt.hist(avg_cof_B_list, bins=20, color='blue', edgecolor='black') # plot the histogram
plt.xlabel('Coefficient of friction, -', fontsize='15', fontweight='bold')
plt.ylabel('Frequency, -', fontsize='15', fontweight='bold')
plt.title('Histogram of COF for B83 40 N', fontsize='18', fontweight='bold')
plt.show()

fig3.savefig('hist_COF_B.png', dpi=500) # save the figure to a file

fig4 = plt.figure() # assign a variable to the figure object
plt.boxplot(avg_cof_B_list, vert=False, showmeans=True) # plot the boxplot
plt.xlabel('Coefficient of friction, -', fontsize='15', fontweight='bold')
plt.title('Boxplot of COF for B83 40 N', fontsize='18', fontweight='bold')
plt.show()

fig4.savefig('box_COF_B.png', dpi=500) # save the figure to a file

# Code lines for material B83 (C)

# Read each .xlsx file containing the data for each dataset of the material B83 with 50 N load

file_C1 = pd.read_excel("B83_50N_1.xlsx") 
file_C2 = pd.read_excel("B83_50N_2.xlsx") 
file_C3 = pd.read_excel("B83_50N_3.xlsx") 

# Assign the dataset to a variable 

time_C1 = file_C1["TIME"] # time in sec. for C1
friction_C1 = file_C1["FF"] # friction force in N  for C1
time_C2 = file_C2["TIME"] # time in sec. for C2
friction_C2 = file_C2["FF"] # friction force in N  for C2
time_C3 = file_C3["TIME"] # time in sec. for C3
friction_C3 = file_C3["FF"] # friction force in N  for C3

# Calculation of the COF for each dataset using the formula: COF = friction force / normal  force (50 N)


cof_C1 = friction_C1 / 50 # COF for  C1 
cof_C2 = friction_C2 / 50 # COF for  C2 
cof_C3 = friction_C3 / 50 # COF for  C3 



# Create an empty list to store the average COF 

avg_cof_C_list= [] 


# Create an empty list to store the average time 

avg_time_C_list= []  

# Average COF 
for i in range(0, len(time_C1)):
    avg_cof_C_list.append((cof_C1[i] + cof_C2[i] + cof_C3[i]) / 3)


# Average time
    avg_time_C_list.append((time_C1[i] + time_C2[i] + time_C3[i]) / 3)


# For material C
mean_cof_C = np.mean(avg_cof_C_list) # calculate the mean
median_cof_C = np.median(avg_cof_C_list) # calculate the median
std_cof_C = np.std(avg_cof_C_list) # calculate the standard deviation
min_cof_C = np.min(avg_cof_C_list) # calculate the minimum
max_cof_C = np.max(avg_cof_C_list) # calculate the maximum

# Print the descriptive statistics for C
print("Descriptive statistics for material C:")
print("Mean: {:.4f}".format(mean_cof_C))
print("Median: {:.4f}".format(median_cof_C))
print("Standard deviation: {:.4f}".format(std_cof_C))
print("Minimum: {:.4f}".format(min_cof_C))
print("Maximum: {:.4f}".format(max_cof_C))

# Save the descriptive statistics to a file
with open('C_descriptive_statistics.txt','w') as f:
 f.write('Descriptive statistics for material C:\n')
 f.write('Mean: {:.4f}\n'.format(mean_cof_C))
 f.write('Median: {:.4f}\n'.format(median_cof_C))
 f.write('Standard deviation: {:.4f}\n'.format(std_cof_C))
 f.write('Minimum: {:.4f}\n'.format(min_cof_C))
 f.write('Maximum: {:.4f}\n'.format(max_cof_C))
 f.close()


# Save the list as a new excel file using pd.DataFrame() and pd.to_excel()

avg_cof_C_df= pd.DataFrame(avg_cof_C_list, columns=["Average Coefficient Of Friction C"]) 
avg_cof_C_df.to_excel("Average_COF_C.xlsx", index=False) 

avg_time_C_df= pd.DataFrame(avg_time_C_list, columns=["Average Time C"]) 
avg_time_C_df.to_excel("Average_time_C.xlsx", index=False) 


# read the data from the two excel files
avg_cof_C_df = pd.read_excel("Average_COF_C.xlsx") 
avg_time_C_df = pd.read_excel("Average_time_C.xlsx")

# concatenate the data from the two files into one DataFrame
merged_df = pd.concat([avg_time_C_df, avg_cof_C_df], axis=1)

# save the new DataFrame to an Excel file
merged_df.to_excel("COF_C.xlsx", index=False)


# load the data from the 'xlsx' file (in our case "pred_COF_C.xlsx")
data = pd.read_excel(r"COF_C.xlsx")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# define a parameter grid with the hyperparameters and their ranges
param_grid = {
   'n_estimators': [50, 100, 200], # number of weak learners
   'learning_rate': [0.1, 0.5, 1.0], # weight of each weak learner
   'loss': ['exponential'], # loss function
   'base_estimator': [DecisionTreeRegressor(max_depth=3)] # base estimator
}

# create an AdaBoost regressor
abr = AdaBoostRegressor(random_state=42) # set random state for reproducibility

# create a grid search object with 5-fold cross-validation
gs = GridSearchCV(abr, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

# fit the grid search on the training data

gs.fit(X_train, y_train)


# print the best parameters and score
print("Best parameters: ", gs.best_params_)
print("Best score: ", gs.best_score_)

# get the best estimator
dt_best = gs.best_estimator_

# predict the coefficient of friction for the test set

y_pred = dt_best.predict(X_test)

# predict the coefficient of friction for the validation set
y_val_pred = dt_best.predict(X_val)

# calculate the R2 score for both sets
r2_test = r2_score(y_test,y_pred)
r2_val = r2_score(y_val,y_val_pred)

# calculate and store the performance metrics for both sets in a file (for "C_performance_metrics.txt")
with open('C_performance_metrics.txt','w') as f:
 f.write('Test set performance metrics:\n')
 f.write('R2 score: {:.4f}\n'.format(r2_test))
 f.write('RMSE: {:.4f}\n'.format(np.sqrt(mean_squared_error(y_test,y_pred))))
 f.write('MSE: {:.4f}\n'.format(mean_squared_error(y_test,y_pred)))
 f.write('MAE: {:.4f}\n'.format(mean_absolute_error(y_test,y_pred)))
 f.write('Validation set performance metrics:\n')
 f.write('R2 score: {:.4f}\n'.format(r2_val))
 f.write('RMSE: {:.4f}\n'.format(np.sqrt(mean_squared_error(y_val,y_val_pred))))
 f.write('MSE: {:.4f}\n'.format(mean_squared_error(y_val,y_val_pred)))
 f.write('MAE: {:.4f}\n'.format(mean_absolute_error(y_val,y_val_pred)))
 f.close()


# Create figure 2
fig2 = plt.figure()



# Shadow effect objects with different transparency and smaller linewidth
pe2 = [patheffects.SimpleLineShadow(offset=(0.5,-0.5), alpha=0.4), patheffects.Normal()]

# Plot of the actual vs predicted COF as a function of time
plt.scatter(X_test[:, 0], y_test,color='cyan',label='Actual test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe2)
plt.scatter(X_test[:, 0], y_pred,color='orange',label='Predicted test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe2)
plt.scatter(X_val[:, 0], y_val,color='green',label='Actual val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe2)
plt.scatter(X_val[:, 0], y_val_pred,color='magenta',label='Predicted val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe2)
plt.xlabel('Time, s', fontsize='15', fontweight='bold')
plt.ylabel('Coefficient of friction, -', fontsize='15', fontweight='bold')
plt.legend(shadow=True, prop={'size':'12'}, loc='lower right')

# x axis limit 
plt.xlim(0 ,450)

# y axis limit 
plt.ylim(0 ,0.3)

# gridlines to the plot
plt.grid(True)


# Add a title
plt.title("B83 50 N", fontsize='18', fontweight='bold')

plt.show()

# Save the plot with dpi=500 in 'png'
fig2.savefig('pred_COF_C.png', dpi=500)


# Close plot material C
plt.close(fig2)

# create a DataFrame from the variables
df = pd.DataFrame({"Actual test": y_test, "Predicted test": y_pred, "Actual val": y_val, "Predicted val": y_val_pred})
# save the DataFrame to an Excel file
df.to_excel("test_val_data_C.xlsx", index=False)


# For material C
fig5 = plt.figure() # assign a variable to the figure object
plt.hist(avg_cof_C_list, bins=20, color='blue', edgecolor='black') # plot the histogram
plt.xlabel('Coefficient of friction, -', fontsize='15', fontweight='bold')
plt.ylabel('Frequency, -', fontsize='15', fontweight='bold')
plt.title('Histogram of COF for B83 50 N', fontsize='18', fontweight='bold')
plt.show()

fig5.savefig('hist_COF_C.png', dpi=500) # save the figure to a file

fig6 = plt.figure() # assign a variable to the figure object
plt.boxplot(avg_cof_C_list, vert=False, showmeans=True) # plot the boxplot
plt.xlabel('Coefficient of friction, -', fontsize='15', fontweight='bold')
plt.title('Boxplot of COF for B83 50 N', fontsize='18', fontweight='bold')
plt.show()

fig6.savefig('box_COF_C.png', dpi=500) # save the figure to a file

# Code lines for (D)

# Read each .xlsx file containing the data for each dataset of the material B83 with 60 N load

file_D1 = pd.read_excel("B83_60N_1.xlsx") 
file_D2 = pd.read_excel("B83_60N_2.xlsx") 
file_D3 = pd.read_excel("B83_60N_3.xlsx") 

# Assign the dataset to a variable 

time_D1 = file_D1["TIME"] # time in sec. for D1
friction_D1 = file_D1["FF"] # friction force in N  for D1
time_D2 = file_D2["TIME"] # time in sec. for D2
friction_D2 = file_D2["FF"] # friction force in N  for D2
time_D3 = file_D3["TIME"] # time in sec. for D3
friction_D3 = file_D3["FF"] # friction force in N  for D3

# Calculation of the COF for each dataset using the formula: COF = friction force / normal  force (60 N)


cof_D1 = friction_D1 / 60 # COF for  D1 
cof_D2 = friction_D2 / 60 # COF for  D2 
cof_D3 = friction_D3 / 60 # COF for  D3 



# Create an empty list to store the average COF 

avg_cof_D_list= [] 


# Create an empty list to store the average time 

avg_time_D_list= []  

# Average COF 
for i in range(0, len(time_D1)):
    avg_cof_D_list.append((cof_D1[i] + cof_D2[i] + cof_D3[i]) / 3)


# Average time
    avg_time_D_list.append((time_D1[i] + time_D2[i] + time_D3[i]) / 3)


# For material D
mean_cof_D = np.mean(avg_cof_D_list) # calculate the mean
median_cof_D = np.median(avg_cof_D_list) # calculate the median
std_cof_D = np.std(avg_cof_D_list) # calculate the standard deviation
min_cof_D = np.min(avg_cof_D_list) # calculate the minimum
max_cof_D = np.max(avg_cof_D_list) # calculate the maximum

# Print the descriptive statistics for D
print("Descriptive statistics for material D:")
print("Mean: {:.4f}".format(mean_cof_D))
print("Median: {:.4f}".format(median_cof_D))
print("Standard deviation: {:.4f}".format(std_cof_D))
print("Minimum: {:.4f}".format(min_cof_D))
print("Maximum: {:.4f}".format(max_cof_D))

# Save the descriptive statistics to a file
with open('D_descriptive_statistics.txt','w') as f:
 f.write('Descriptive statistics for material D:\n')
 f.write('Mean: {:.4f}\n'.format(mean_cof_D))
 f.write('Median: {:.4f}\n'.format(median_cof_D))
 f.write('Standard deviation: {:.4f}\n'.format(std_cof_D))
 f.write('Minimum: {:.4f}\n'.format(min_cof_D))
 f.write('Maximum: {:.4f}\n'.format(max_cof_D))
 f.close()


# Save the list as a new excel file using pd.DataFrame() and pd.to_excel()

avg_cof_D_df= pd.DataFrame(avg_cof_D_list, columns=["Average Coefficient Of Friction D"]) 
avg_cof_D_df.to_excel("Average_COF_D.xlsx", index=False) 

avg_time_D_df= pd.DataFrame(avg_time_D_list, columns=["Average Time D"]) 
avg_time_D_df.to_excel("Average_time_D.xlsx", index=False) 


# read the data from the two excel files
avg_cof_D_df = pd.read_excel("Average_COF_D.xlsx") 
avg_time_D_df = pd.read_excel("Average_time_D.xlsx")

# concatenate the data from the two files into one DataFrame
merged_df = pd.concat([avg_time_D_df, avg_cof_D_df], axis=1)

# save the new DataFrame to an Excel file
merged_df.to_excel("COF_D.xlsx", index=False)


# load the data from the 'xlsx' file (in our case "pred_COF_B.xlsx")
data = pd.read_excel(r"COF_D.xlsx")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# split the data into training, validation, and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

# define a parameter grid with the hyperparameters and their ranges
param_grid = {
   'n_estimators': [50, 100, 200], # number of weak learners
   'learning_rate': [0.1, 0.5, 1.0], # weight of each weak learner
   'loss': ['exponential'], # loss function
   'base_estimator': [DecisionTreeRegressor(max_depth=3)] # base estimator
}

# create an AdaBoost regressor
abr = AdaBoostRegressor(random_state=42) # set random state for reproducibility

# create a grid search object with 5-fold cross-validation
gs = GridSearchCV(abr, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

# fit the grid search on the training data

gs.fit(X_train, y_train)


# print the best parameters and score
print("Best parameters: ", gs.best_params_)
print("Best score: ", gs.best_score_)

# get the best estimator
dt_best = gs.best_estimator_

# predict the coefficient of friction for the test set

y_pred = dt_best.predict(X_test)

# predict the coefficient of friction for the validation set
y_val_pred = dt_best.predict(X_val)

# calculate the R2 score for both sets
r2_test = r2_score(y_test,y_pred)
r2_val = r2_score(y_val,y_val_pred)

# calculate and store the performance metrics for both sets in a file (for "D_performance_metrics.txt")
with open('D_performance_metrics.txt','w') as f:
 f.write('Test set performance metrics:\n')
 f.write('R2 score: {:.4f}\n'.format(r2_test))
 f.write('RMSE: {:.4f}\n'.format(np.sqrt(mean_squared_error(y_test,y_pred))))
 f.write('MSE: {:.4f}\n'.format(mean_squared_error(y_test,y_pred)))
 f.write('MAE: {:.4f}\n'.format(mean_absolute_error(y_test,y_pred)))
 f.write('Validation set performance metrics:\n')
 f.write('R2 score: {:.4f}\n'.format(r2_val))
 f.write('RMSE: {:.4f}\n'.format(np.sqrt(mean_squared_error(y_val,y_val_pred))))
 f.write('MSE: {:.4f}\n'.format(mean_squared_error(y_val,y_val_pred)))
 f.write('MAE: {:.4f}\n'.format(mean_absolute_error(y_val,y_val_pred)))
 f.close()


# Create figure 3
fig3 = plt.figure()



# Shadow effect objects with different transparency and smaller linewidth
pe3 = [patheffects.SimpleLineShadow(offset=(0.5,-0.5), alpha=0.4), patheffects.Normal()]

# Plot of the actual vs predicted COF as a function of time
plt.scatter(X_test[:, 0], y_test,color='cyan',label='Actual test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe3)
plt.scatter(X_test[:, 0], y_pred,color='orange',label='Predicted test', linewidth=1,alpha=0.9,zorder=1,path_effects=pe3)
plt.scatter(X_val[:, 0], y_val,color='green',label='Actual val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe3)
plt.scatter(X_val[:, 0], y_val_pred,color='magenta',label='Predicted val', linewidth=1,alpha=0.9,zorder=1,path_effects=pe3)
plt.xlabel('Time, s', fontsize='15', fontweight='bold')
plt.ylabel('Coefficient of friction, -', fontsize='15', fontweight='bold')
plt.legend(shadow=True, prop={'size':'12'}, loc='lower right')

# x axis limit 
plt.xlim(0 ,450)

# y axis limit 
plt.ylim(0 ,0.3)

# gridlines to the plot
plt.grid(True)


# Add a title
plt.title("B83 60 N", fontsize='18', fontweight='bold')

plt.show()

# Save the plot with dpi=500 in 'png'
fig3.savefig('pred_COF_D.png', dpi=500)


# Close plot material D
plt.close(fig3)

# create a DataFrame from the variables
df = pd.DataFrame({"Actual test": y_test, "Predicted test": y_pred, "Actual val": y_val, "Predicted val": y_val_pred})
# save the DataFrame to an Excel file
df.to_excel("test_val_data_D.xlsx", index=False)


# For material D
fig7 = plt.figure() # assign a variable to the figure object
plt.hist(avg_cof_D_list, bins=20, color='blue', edgecolor='black') # plot the histogram
plt.xlabel('Coefficient of friction, -', fontsize='15', fontweight='bold')
plt.ylabel('Frequency, -', fontsize='15', fontweight='bold')
plt.title('Histogram of COF for B83 60 N', fontsize='18', fontweight='bold')
plt.show()

fig7.savefig('hist_COF_D.png', dpi=500) # save the figure to a file

fig8 = plt.figure() # assign a variable to the figure object
plt.boxplot(avg_cof_D_list, vert=False, showmeans=True) # plot the boxplot
plt.xlabel('Coefficient of friction, -', fontsize='15', fontweight='bold')
plt.title('Boxplot of COF for B83 60 N', fontsize='18', fontweight='bold')
plt.show()

fig8.savefig('box_COF_D.png', dpi=500) # save the figure to a file




# Bar plot of the average COF with the std for the materials

# load the data from the three Excel files
data_B = pd.read_excel(r"COF_B.xlsx")
data_C = pd.read_excel(r"COF_C.xlsx")
data_D = pd.read_excel(r"COF_D.xlsx")

# extract the mean COF values for each material
mean_cof_B = data_B["Average Coefficient Of Friction B"].mean()
mean_cof_C = data_C["Average Coefficient Of Friction C"].mean()
mean_cof_D = data_D["Average Coefficient Of Friction D"].mean()

# extract the standard deviation values for each material
std_cof_B = data_B["Average Coefficient Of Friction B"].std()
std_cof_C = data_C["Average Coefficient Of Friction C"].std()
std_cof_D = data_D["Average Coefficient Of Friction D"].std()

# create a list of materials
materials = ["B83 40 N", "B83 50 N", "B83 60 N"]

# create a list of mean COF values
mean_cof = [mean_cof_B, mean_cof_C, mean_cof_D]

# create a list of standard deviation values
std_cof = [std_cof_B, std_cof_C, std_cof_D]

# create a bar plot of the mean COF values with standard deviation bars for each material
plt.errorbar(materials, mean_cof, yerr=std_cof, fmt='-o')


# add labels and title
plt.xlabel('Material', fontsize='15', fontweight='bold')
plt.ylabel('Mean Coefficient Of Friction', fontsize='15', fontweight='bold')
plt.title('Average COF', fontsize='18', fontweight='bold')

# get the current figure object
fig9 = plt.gcf()

# show the plot
plt.show()

# save the figure to a file named 'bar_plot.png' with 500 dpi resolution
fig9.savefig('bar_plot.png', dpi=500) # save the figure to a file
