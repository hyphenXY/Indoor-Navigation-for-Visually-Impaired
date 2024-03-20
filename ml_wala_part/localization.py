#!/usr/bin/env python
import os
from IPython.display import display
from itertools import product
import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import accuracy_score
from sklearn.gaussian_process.kernels import RationalQuadratic
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, RationalQuadratic, ExpSineSquared


# %matplotlib notebook


# Directory containing beacon files
def GroundTruthData(directory_gt, floor, test_location, num_files):
    # Initialize an empty list to hold dataframes
    dataframes_gt = []
    # Loop through each beacon file
    for filename in os.listdir(directory_gt):
        # print(filename)
        file_path = os.path.join(directory_gt, filename)
        df = pd.read_csv(file_path)
        df.rename(columns={'rssi_mean': filename.split('.csv')[0]}, inplace=True)
        df = df.drop(['rssi_std'], axis=1)  # Drop 'rssi_std' column
        dataframes_gt.append(df)  # Append dataframe to the list

    dataframe_gt = pd.concat([df.set_index('location') for df in dataframes_gt], axis=1, join='outer').reset_index()
    # Merge all dataframes on 'location'
    # dataframe_gt = pd.concat([df.set_index('location') for df in dataframes_gt], axis=1, join='inner').reset_index()
    # print("==============================================")
    # for column in dataframe_gt.columns[1:]:
    #     mean_value = dataframe_gt[column].mean()
    #     dataframe_gt[column].fillna(mean_value, inplace=True)
    dataframe_gt.fillna('-95', inplace=True)
    # print(dataframe_gt.round(0))
    dataframe_gt.to_csv(floor + '/all_scan_sample.csv', index=False)

    ### Randomly sampled 6 test location out of total 29 location and 3 outlier = 32 points####
    test_data = dataframe_gt.loc[dataframe_gt['location'].isin(test_location)]
    test_data = test_data.sort_values(by=['location'])
    temp = test_data.location.str.split(",", n=-1, expand=True)
    # print(dataframe_gt)
    test_data['cordX'] = temp[0].astype(int)
    test_data['cordY'] = temp[1].astype(int)
    # print(data.head())
    X_gt = test_data.iloc[:, 1:num_files]
    y1_gt = test_data['cordX']
    y2_gt = test_data['cordY']
    # format : location 	7CD9F4011469  7CD9F4012133  7CD9F4027CE6  7CD9F4034BCB  7CD9F40399BD  7CD9F403AFF6  7CD9F4071256  7CD9F40966EA	  cordX	  cordY
    test_data.head()
    test_data.shape

    return X_gt, y1_gt, y2_gt


### beacons_pd folder include predicted beacons rssi data ########
def GPPredictedData(directory_pd, num_files):
    dataframes_pd = []
    for filename in os.listdir(directory_pd):
        file_path = os.path.join(directory_pd, filename)
        # Read CSV file
        df = pd.read_csv(file_path)
        # Rename column
        df.rename(columns={'rssi_mean': filename.split('.csv')[0]}, inplace=True)
        # Drop 'rssi_std' column
        df = df.drop(['rssi_std'], axis=1)
        # Append dataframe to the list
        dataframes_pd.append(df)
    # Merge all dataframes on 'location'
    dataframes_pd = pd.concat([df.set_index('location') for df in dataframes_pd], axis=1, join='inner').reset_index()
    new = dataframes_pd.location.str.split(",", n=-1, expand=True)
    dataframes_pd['cordX'] = new[0].astype(int)
    dataframes_pd['cordY'] = new[1].astype(int)
    X_pd = dataframes_pd.iloc[:, 1:num_files]
    y1_pd = dataframes_pd['cordX']
    y2_pd = dataframes_pd['cordY']
    # format : location 	GP_HBD_7CD9F4011469  GP_HBD_7CD9F4012133  ...  GP_HBD_7CD9F403AFF6  GP_HBD_7CD9F4071256  GP_HBD_7CD9F40966EA	  cordX	  cordY
    dataframes_pd.head()
    dataframes_pd.shape

    return X_pd, y1_pd, y2_pd


#################### KNN training for X-axis ####################
def knnXaxis(X_train, y1_train, X_test, y1_test, y1_gt):
    print("---------KNN training for X-axis result---------")
    # print(X_test, y1_test)
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    # print(X_train, y1_train)
    knn_classifier.fit(X_train, y1_train.values.ravel())
    # test using GP predicted test dataset
    y1_pred = knn_classifier.predict(X_test)
    # print(y1_test, y1_pred)
    # knn_accuracy=accuracy_score(y1_test, y1_pred)
    # print('\nThe Models Accuracy is', knn_accuracy)
    MAE1 = mean_absolute_error(y1_pred, y1_test)
    print("Predicted data knn-mae-X-axis :\t", MAE1)
    # test using ground truth test dataset
    # print("Inputs==========",X_gt)
    y1_pred_gt = knn_classifier.predict(X_gt)
    # print("Outputs======",y1_pred_gt)
    # # knn_accuracy=accuracy_score(y1_gt, y1_pred_gt)
    # # print('\nThe Models Accuracy for ground truth values, is', knn_accuracy)
    # print(y1_gt,y1_pred_gt)
    MAE2 = mean_absolute_error(y1_pred_gt, y1_gt)
    print("Ground truth data knn-mae-X-axis :\t", MAE2)

    return MAE1, MAE2, y1_pred, y1_pred_gt


#################### KNN training for Y-axis ####################
def knnYaxis(X_train, y2_train, X_test, y2_test, y2_gt):
    print("---------KNN training for Y-axis result---------")
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    knn_classifier.fit(X_train, y2_train.values.ravel())
    # test using GP predicted test dataset
    y2_pred = knn_classifier.predict(X_test)
    # knn_accuracy=accuracy_score(y2_test, y2_pred)
    # print('\nThe Models Accuracy is', knn_accuracy)

    MAE3 = mean_absolute_error(y2_pred, y2_test)
    print("Predicted data knn-mae-Y-axis :\t", MAE3)
    # test using ground truth test dataset
    y2_pred_gt = knn_classifier.predict(X_gt)
    MAE4 = mean_absolute_error(y2_pred_gt, y2_gt)
    # print(y2_gt,y2_pred_gt)
    print("Ground truth data knn-mae-Y-axis :\t", MAE4)

    return MAE3, MAE4, y2_pred, y2_pred_gt


########## KNN - Localization performance #########################
########### Part 1: Result on GP predicted test dataset ##################
def knnLocalizationPredictedData(y1_test, y2_test, X_test, y1_pred, y2_pred, path_3):
    r1 = pd.concat([y1_test, y2_test, X_test], axis=1)
    # r1.rename(columns={'cordX': 'cordXold', 1: 'cordYold'}, inplace=True)
    # r1.drop(columns=["cordX", "cordY"], inplace=True)
    r2 = pd.concat([pd.DataFrame({'SP_cordX': y1_pred}), pd.DataFrame({'SP_cordY': y2_pred})], axis=1)
    result = pd.concat([r1, r2.set_index(r1.index)], axis=1)
    display(result)
    # euclidean distance error in a grid world
    # result = result.rename(columns={0: 'cordXold', 1: 'cordYold'}, inplace=True)
    result['dist_localization_error_pd'] = np.round_(
        np.sqrt((result['cordX'] - result['SP_cordX']) ** 2 + (result['cordY'] - result['SP_cordY']) ** 2), 2)
    result['loc'] = (result['cordX'].map(str)).str.cat(result['cordY'].map(str), sep=",")
    result['Pred_loc'] = (result['SP_cordX'].map(str)).str.cat(result['SP_cordY'].map(str), sep=",")
    result = result[['loc', 'Pred_loc', 'dist_localization_error_pd']]
    # print("Part 1: Result on GP predicted test dataset ==========")

    print("Localziation_mse_knn_pd_predicted", np.mean(result['dist_localization_error_pd']) * 0.305, "m")
    # print("Localization error WRT Predicted data", result['dist_localization_error_pd']*0.305, "m")
    result.to_csv(path_3 + '/knn_result_pd_predicted.csv', index=False)

    return np.mean(result['dist_localization_error_pd'])


########## Part 2: Result on Ground truth test dataset ###################
def knnLocalizationGroundTruthData(y1_gt, y2_gt, X_gt, y1_pred_gt, y2_pred_gt, path_4):
    r1 = pd.concat([y1_gt, y2_gt, X_gt], axis=1)
    r2 = pd.concat([pd.DataFrame({'GPP_cordX': y1_pred_gt}), pd.DataFrame({'GPP_cordY': y2_pred_gt})], axis=1)
    result = pd.concat([r1, r2.set_index(r1.index)], axis=1)
    print(result)
    # euclidean distance error in a grid world
    result['dist_localization_error_gt'] = np.round_(
        np.sqrt((result['cordX'] - result['GPP_cordX']) ** 2 + (result['cordY'] - result['GPP_cordY']) ** 2), 2)
    result['loc'] = (result['cordX'].map(str)).str.cat(result['cordY'].map(str), sep=",")
    result['Pred_loc'] = (result['GPP_cordX'].map(str)).str.cat(result['GPP_cordY'].map(str), sep=",")
    result = result[['loc', 'Pred_loc', 'dist_localization_error_gt']]
    # print("Part 2: Result on Ground truth test dataset ==========")

    print("Localziation_mse_knn_gt_predicted", np.mean(result['dist_localization_error_gt']) * 0.305, "m")
    # print("Localization error WRT Ground truth data", result['dist_localization_error_gt']*0.305, "m")
    result.to_csv(path_4 + '/knn_result_gt_predicted.csv', index=False)

    return np.mean(result['dist_localization_error_gt'])


#################################################################
########### Random Forest Classification X-axis #################
#################################################################
def rfXaxis(X_train, y1_train, X_test, y1_test, X_gt, y1_gt):
    print("---------RF training for X-axis result---------")

    rf_classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
    rf_classifier.fit(X_train, y1_train.values.ravel())

    #### using GP predicted test dataset
    y1_pred = rf_classifier.predict(X_test)
    # rf_accuracy=accuracy_score(y1_test, y1_pred)
    # print('\nThe Models Accuracy is', rf_accuracy)
    MAE5 = mean_absolute_error(y1_pred, y1_test)
    print("Predicted data rf-mae-X-axis :\t", MAE5)

    #### using ground truth test dataset
    y1_pred_gt = rf_classifier.predict(X_gt)
    MAE6 = mean_absolute_error(y1_pred_gt, y1_gt)
    print("Ground truth data rf-mae-X-axis :\t", MAE6)

    return MAE5, MAE6, y1_pred, y1_pred_gt


#################################################################
########### Random Forest Classification Y-axis #################
#################################################################
def rfYaxis(X_train, y2_train, X_test, y2_test, X_gt, y2_gt):
    print("---------RF training for Y-axis result---------")
    rf_classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
    rf_classifier.fit(X_train, y2_train.values.ravel())

    #### using GP predicted test dataset
    y2_pred = rf_classifier.predict(X_test)
    # rf_accuracy=accuracy_score(y2_test, y2_pred)
    # print('\nThe Models Accuracy is', rf_accuracy)
    MAE7 = mean_absolute_error(y2_pred, y2_test)
    print("Predicted data rf-mae-Y-axis :\t", MAE7)

    ##### using ground truth test dataset
    y2_pred_gt = rf_classifier.predict(X_gt)
    y2_pred_gt = np.round_(y2_pred_gt, decimals=2)
    # rf_accuracy=accuracy_score(y2_gt, y2_pred_gt)
    # print('\nThe Models Accuracy for ground truth values, is', rf_accuracy)
    MAE8 = mean_absolute_error(y2_pred_gt, y2_gt)
    print("Ground truth data rf-mae-Y-axis :\t", MAE8)

    return MAE7, MAE8, y2_pred, y2_pred_gt


##################################################################
########## RF - Localization performance #########################
##################################################################

def rfLocalizationPredictedData(y1_test, y2_test, X_test, y1_pred, y2_pred):
    ########### Part 1: Result on GP predicted test dataset ###########
    r1 = pd.concat([y1_test, y2_test, X_test], axis=1)
    r2 = pd.concat([pd.DataFrame({'SP_cordX': y1_pred}), pd.DataFrame({'SP_cordY': y2_pred})], axis=1)
    result = pd.concat([r1, r2.set_index(r1.index)], axis=1)
    result['dist_localization_error_pd'] = np.round_(
        np.sqrt((result['cordX'] - result['SP_cordX']) ** 2 + (result['cordY'] - result['SP_cordY']) ** 2), 2)
    # print("Part 1: Result on GP predicted test dataset ==========")
    # display(result)
    print("Localziation_mse_rf_pd_predicted", np.mean(result['dist_localization_error_pd']))
    result.to_csv('Third/results/rf_result_pd_predicted.csv', index=False)

    return np.mean(result['dist_localization_error_pd'])


def rfLocalizationGroundTruthData(y1_gt, y2_gt, X_gt, y1_pred_gt, y2_pred_gt):
    ########## Part 2: Result on Ground truth test dataset ###################
    r1 = pd.concat([y1_gt, y2_gt, X_gt], axis=1)
    r2 = pd.concat([pd.DataFrame({'GPP_cordX': y1_pred_gt}), pd.DataFrame({'GPP_cordY': y2_pred_gt})], axis=1)
    result = pd.concat([r1, r2.set_index(r1.index)], axis=1)
    result['dist_localization_error_gt'] = np.round_(
        np.sqrt((result['cordX'] - result['GPP_cordX']) ** 2 + (result['cordY'] - result['GPP_cordY']) ** 2), 2)
    result['loc'] = (result['cordX'].map(str)).str.cat(result['cordY'].map(str), sep=",")
    result['Pred_loc'] = (result['GPP_cordX'].map(str)).str.cat(result['GPP_cordY'].map(str), sep=",")
    result = result[['loc', 'Pred_loc', 'dist_localization_error_gt']]
    # print("Part 2: Result on Ground truth test dataset ==========")
    # display(result)
    print("Localziation_mse_rf_gt_predicted", np.mean(result['dist_localization_error_gt']))
    result.to_csv('Third/results/rf_result_gt_predicted.csv', index=False)

    return np.mean(result['dist_localization_error_gt'])


floor = 'Third'
# floor = 'Ground'
path_1 = floor + '/beacons_gt/'

test_location_third = ["50,33", "27,33", "39,54", "67,57", "74,55", "110,54", "118,54", "118,47", "92,25", "51,56",
                       "62,55", "122,54", "71,58"]
test_location_ground = ["134,99", "36,59", "153,121", "76,80", "47,78", "119,80", "80,78",
                        "33,83", "168,118", "165,97", "47,78", "17,78"]

num_files = len(os.listdir(path_1)) + 1  # Count the number of files
if floor == "Ground":
    test_location = test_location_ground
else:
    test_location = test_location_third

print(test_location)
X_gt, y1_gt, y2_gt = GroundTruthData(path_1, floor, test_location, num_files)
# print(X_gt)
path_2 = floor + '/beacons_pd/'
num_files = len(os.listdir(path_2)) + 1  # Count the number of files
X_pd, y1_pd, y2_pd = GPPredictedData(path_2, num_files)
print(num_files)

# #### Split the data set for training and testing ###########
X_train, X_test, y1_train, y1_test = train_test_split(X_pd, y1_pd, test_size=0.2, random_state=42)
mae_pd_x, mae_gt_x, y1_pred, y1_pred_gt = knnXaxis(X_train, y1_train, X_test, y1_test, y1_gt)

# print(X_pd)
# #### Split the data set for training and testing ###########
X_train, X_test, y2_train, y2_test = train_test_split(X_pd, y2_pd, test_size=0.2, random_state=42)
mae_pd_y, mae_gt_y, y2_pred, y2_pred_gt = knnYaxis(X_train, y2_train, X_test, y2_test, y2_gt)
# print(y1_gt.to_frame()['cordX'], y2_gt.to_frame()['cordY'])
y1_y2_gt = pd.concat([y1_gt.to_frame(), y2_gt.to_frame()], axis=1)
# print(y1_y2_gt)
y1_y2_pd_gt = pd.concat([pd.DataFrame(y1_pred_gt), pd.DataFrame(y2_pred_gt)], axis=1)
y1_y2_pd_gt.rename(columns={'0': 'cordX', '0': 'cordY'})
# print(pd.DataFrame(y1_pred_gt))
# print(y1_y2_pd_gt)
# print(pd.concat([y1_y2_gt, y1_y2_pd_gt], axis=1))
print("---------KNN Localization performance evaluatiion ---------")
path_3 = floor + '/results'
path_4 = floor + '/results'
mse_knn_pd = knnLocalizationPredictedData(y1_test, y2_test, X_test, y1_pred, y2_pred, path_3)
mse_knn_gt = knnLocalizationGroundTruthData(y1_gt, y2_gt, X_gt, y1_pred_gt, y2_pred_gt, path_4)

# MAE5, MAE6, y1_pred, y1_pred_gt  = rfXaxis(X_train,y1_train, X_test,y1_test, X_gt, y1_gt)
# MAE7, MAE8, y2_pred, y2_pred_gt =  rfYaxis(X_train, y2_train, X_test, y2_test, X_gt, y2_gt)
# print("---------RF Localization performance evaluatiion ---------" )
# mse_rf_pd = rfLocalizationPredictedData(y1_test, y2_test,X_test,y1_pred,y2_pred)
# mse_rf_gt = rfLocalizationGroundTruthData(y1_gt, y2_gt,X_gt, y1_pred_gt,y2_pred_gt)


#################################################################
################# Gaussian Process Regression ####################
#################################################################
#### Define hybrid GP kernel function
# long_term_trend_kernel = 50.0 ** 2 * RBF(length_scale=50.0)
# seasonal_kernel = (
#     2.0 ** 2
#     * RBF(length_scale=100.0)
#     * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
# )

# irregularities_kernel = 0.5 ** 2 * RationalQuadratic(length_scale=1.0, alpha=1.0)

# noise_kernel = 0.1 ** 2 * RBF(length_scale=0.1) + WhiteKernel(
#     noise_level=0.1 ** 2, noise_level_bounds=(1e-5, 1e5)
#     )

# rssi_kernel = long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel

'''
#################################################################
########### Gaussian Process Regression X-axis #################
#################################################################
print("---------GP regression for X-axis result---------" )

RQ_kernel = 0.5 ** 2 * RationalQuadratic(length_scale=0.2, alpha=1.0)
gp = GaussianProcessRegressor(kernel=rssi_kernel, n_restarts_optimizer=10,normalize_y=False)
gp.fit(X_train, y1_train.values.ravel())
print("Done!!!!!")
# _check_optimize_result("lbfgs", opt_res)

#### using GP predicted test dataset
y1_pred, std1 = gp.predict(X_test, return_std=True)
y1_pred = np.round_(y1_pred, decimals = 2)
std1 = np.round_(std1, decimals = 2)
# print("y_pred mean\n",np.round_(y1_pred,2),"\nstd_dev\n",std1)
MAE9 = mean_absolute_error(y1_pred,y1_test)
print("Predicted data gp-mae-X-axis :\t",MAE9)

###### Using ground truth test dataset 
y1_pred_gt,std2=gp.predict(X_gt, return_std=True)
y1_pred_gt = np.round_(y1_pred_gt, decimals = 2)
MAE10 = mean_absolute_error(y1_pred_gt,y1_gt)
print("Ground truth data gp-mae-X-axis :\t",MAE10)


#################################################################
########### Gaussian Process Regression Y-axis #################
#################################################################
print("---------GP regression for Y-axis result---------" )
# exp_kernel = 0.5 ** 2 * RBF(length_scale=0.2)
gp = GaussianProcessRegressor(kernel=RQ_kernel, n_restarts_optimizer=15,normalize_y=False)
gp.fit(X_train, y2_train.values.ravel())

#### using GP predicted test dataset
y2_pred, std2 = gp.predict(X_test, return_std=True)
y2_pred = np.round_(y2_pred, decimals = 2)
std2 = np.round_(std2, decimals = 2)
# print("y_pred mean\n",np.round_(y1_pred,2),"\nstd_dev\n",std1)
MAE11 = mean_absolute_error(y2_pred,y2_test)
print("Predicted data gp-mae-Y-axis :\t",MAE11)

###### Using ground truth test dataset 
y2_pred_gt,std2=gp.predict(X_gt, return_std=True)
y2_pred_gt = np.round_(y2_pred_gt, decimals = 2)
# print(y1_pred)
# print(y2_gt)
# print(y1_pred_g2)
MAE12 = mean_absolute_error(y2_pred_gt,y2_gt)
print("Ground truth data gp-mae-Y-axis :\t",MAE12)

##################################################################
########## GP - Localization performance #########################
##################################################################
print("---------GP Localization performance evaluatiion ---------" )

########### Part 1: Result on GP predicted test dataset ########### 
r1 = pd.concat([y1_test, y2_test,X_test], axis=1)
r2 = pd.concat([pd.DataFrame({'SP_cordX':y1_pred}),pd.DataFrame({'SP_cordY':y2_pred})], axis=1)
result = pd.concat([r1,r2.set_index(r1.index)], axis=1)
result['dist_localization_error_pd']  = np.round_(np.sqrt((result['cordX'] - result['SP_cordX'])**2 + (result['cordY'] - result['SP_cordY'])**2),2)
# print("Part 1: Result on predicted test dataset ==========")
# display(result)
print("Localziation_mse_gp_pd_predicted", np.mean(result['dist_localization_error_pd'] ))
result.to_csv('gp_result_pd_predicted.csv', index=False)

########### Part 2: Result on GP ground truth test dataset ########### 
r1 = pd.concat([y1_gt, y2_gt,X_gt], axis=1)
r2 = pd.concat([pd.DataFrame({'GPP_cordX':y1_pred_gt}),pd.DataFrame({'GPP_cordY':y2_pred_gt})], axis=1)
result = pd.concat([r1,r2.set_index(r1.index)], axis=1)
result['dist_localization_error_gt']  = np.round_(np.sqrt((result['cordX'] - result['GPP_cordX'])**2 + (result['cordY'] - result['GPP_cordY'])**2),2)
result['loc'] = (result['cordX'].map(str)).str.cat(result['cordY'].map(str),sep=",")
result['Pred_loc'] = (result['GPP_cordX'].map(str)).str.cat(result['GPP_cordY'].map(str),sep=",")
result = result[['loc','Pred_loc','dist_localization_error_gt']]
# print("Part 2: Result on Ground truth test dataset ==========")
# display(result)
print("Localziation_mse_gp_gt_predicted", np.mean(result['dist_localization_error_gt'] ))
result.to_csv('gp_result_gt_predicted.csv', index=False)

# %%
'''