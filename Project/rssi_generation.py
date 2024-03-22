import os
import re
from fileinput import filename
import sys
from itertools import product
import numpy as np
from sklearn.metrics import mean_absolute_error
import pandas as pd
from matplotlib import cm as CM
from matplotlib import mlab as ML
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel as C,
    WhiteKernel,
    RationalQuadratic,
    ExpSineSquared,
)

# get_ipython().run_line_magic('matplotlib', 'notebook')


def testAndTrainData(df1, df2):
    #######training points######
    tmp1 = df1["location"].to_numpy()
    y1 = df1["rssi_mean"]
    y_train = y1.to_numpy()  # rssi traing data
    X1 = []
    for i in tmp1:
        t = list(map(int, i.split(",")))
        X1.append(t)
    X_train = np.array(X1)
    # print(X_train) # training location
    #######testing points#######
    tmp2 = df2["location"].to_numpy()
    y2 = df2["rssi_mean"]
    y_test = y2.to_numpy()
    X2 = []
    for i in tmp2:
        t = list(map(int, i.split(",")))
        X2.append(t)
    X_test = np.array(X2)

    return X_train, y_train, X_test, y_test


####### kernal used in Gaussian Process Regression #######
def hybridGPKernal():
    long_term_trend_kernel = 50.0**2 * RBF(length_scale=50.0)
    seasonal_kernel = (
        2.0**2
        * RBF(length_scale=100.0)
        * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds="fixed")
    )

    irregularities_kernel = 0.5**2 * RationalQuadratic(length_scale=1.0, alpha=1.0)

    noise_kernel = 0.1**2 * RBF(length_scale=0.1) + WhiteKernel(
        noise_level=0.1**2, noise_level_bounds=(1e-5, 1e5)
    )

    rssi_kernel = (
        long_term_trend_kernel + seasonal_kernel + irregularities_kernel + noise_kernel
    )

    return rssi_kernel


########### Genrating likelihood RSSI data-set for each beacon  ##########
floor = "Project/ground"
folder_path = floor + "/beacons_gt/"

test_location_third = [
    "50,33",
    "54,38",
    "40,44",
    "47,52",
    "94,58",
    "33,57",
    "74,55",
    "72,76",
    "89,41",
    "108,32",
    "92,25",
    "149,23",
    "142,72",
]
test_location_ground = [
    "134,99",
    "25,80",
    "49,80",
    "36,59",
    "63,92",
    "76,80",
    "47,78",
    "93,78",
    "119,80",
    "116,53",
    "144,66",
    "144,84",
    "165,97",
]

if floor == "Project/ground":
    test_location = test_location_ground
else:
    test_location = test_location_third

for filename in os.listdir(folder_path):
    print(filename)
    if filename.endswith(".csv"):
        df = pd.read_csv(folder_path + filename)
        v = filename.split(".csv")[0]
        # print(v,filename)

    df1 = df.loc[~df["location"].isin(test_location)]
    df2 = df.loc[df["location"].isin(test_location)]
    df2 = df2.sort_values(by=["location"])

    X_train, y_train, X_test, y_test = testAndTrainData(df1, df2)
    x1 = np.linspace(1, 188, 188).astype(int)  # p
    x2 = np.linspace(1, 130, 130).astype(int)  # q
    # print("Training data==================", X_train, y_train)
    # print("Testing data==================", X_test, y_test)
    # Gaussian (or RBF kernal) kernal
    kernel = C(1.0, (1e-1, 1e1)) * RBF([1, 1], (1e-2, 1e2))
    #### Trained with Hybrid Kernal
    rssi_kernel = hybridGPKernal()
    gp = GaussianProcessRegressor(
        kernel=rssi_kernel, n_restarts_optimizer=10, normalize_y=False
    )

    #### Trained with Gaussian (or RBF kernal) kernal
    # gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,normalize_y=False)

    gp.fit(X_train, y_train)
    # gp.fit(X,y)
    # print("param",gp.kernel_.get_params())
    # print(gp.get_params(deep=True))

    X0p, X1p = np.meshgrid(x1, x2)
    # print(X0p.shape[0])

    ####### RSSI mean and standard deviation prediction for input space ########
    data = []
    Zp_mean = []
    Zp_std = []
    # temp_a =[]
    for i in range(X0p.shape[0]):
        tmp_mean = []
        tmp_std = []
        for j in range(X1p.shape[1]):
            a = gp.predict([(X0p[i, j], X1p[i, j])], return_std=True)
            # temp_a.append(a[0][0])
            # print((X0p[i,j] ,X1p[i,j]), a[0][0])
            # if not((X0p[i,j]>=47 and X0p[i,j]<=63 and X1p[i,j]>=54 and X1p[i,j]<=63) or ((X0p[i,j]<=36 or X0p[i,j]>=75) and X1p[i,j]>=54)) :
            data.append(
                [
                    str(X0p[i, j]) + "," + str(X1p[i, j]),
                    round(a[0][0], 2),
                    round(a[1][0], 2),
                ]
            )
            tmp_mean.append(round(a[0][0], 2))
            tmp_std.append(round(a[1][0], 2))
        Zp_mean.append(tmp_mean)
        Zp_std.append(tmp_std)

    # print(len(temp_a))
    Zp_mean = np.asarray(Zp_mean)
    Zp_mean = np.array(Zp_mean).T
    Zp_mean = Zp_mean.T
    # print(Zp_mean)

    Zp_std = np.asarray(Zp_std)
    Zp_std = np.array(Zp_std).T
    Zp_std = Zp_std.T
    # print(Zp_std)

    ######## Likelihood RSSI data-set generated for each beacon ###########
    df1 = pd.DataFrame(data, columns=["location", "rssi_mean", "rssi_std"])
    # df1.to_csv('pd/'+'GP_HBD_'+v[0], index=False)
    df1.to_csv("Project/ground/beacons_pd/" + v + ".csv", index=False)
    # print(df1.shape)

    ##### mean_absolute_error on testing point #####
    # print(X_test)
    y_pred, std = gp.predict(X_test, return_std=True)
    y_pred = np.round_(y_pred, decimals=2)
    # print(y_pred)
    std = np.round_(std, decimals=2)
    # print("X_test\n", X_test)
    # print("y_test\n", y_test)
    # print("y_pred mean\n",np.round_(y_pred,2),"\nstd dev\n",std)

    ################################################
    MAE = mean_absolute_error(y_pred, y_test)
    print(" mean_absolute_error_ground :\t", round(MAE, 2))

    ################ plot functions  #####################
    x = X0p.ravel()
    y = X1p.ravel()

    ####### Plotting the mean RSSI value of the input space ########
    z = Zp_mean.ravel()
    ####### Plotting the standard deviation RSSI value of the input space ########
    # z = Zp_std.ravel()
    # gridsize=70
    # plt.subplot(111)

    # # if 'bins=None', then color of each hexagon corresponds directly to its count
    # # 'C' is optional--it maps values to x-y coordinates; if 'C' is None (default) then
    # # the result is a pure 2D histogram
    # plt.hexbin(x, y, C=z, gridsize=gridsize, cmap=CM.jet, bins=None)
    # plt.axis([x.min(), x.max(), y.min(), y.max()])
    # plt.xlabel('x_axis')
    # plt.ylabel('y_axis')
    # plt.ylim(max(y), min(y))
    # cb = plt.colorbar()
    # cb.set_label('mean_rssi')
    # # plt.savefig("plots/"+v[0].split('.csv')[0], dpi=2000)
    # plt.savefig("plots_ground/"+v, dpi=2000)
    # # cb.set_label('std_rssi')
    # # print(v[0].split('.csv')[0])
    # # # plt.savefig("Standard deviation of GP prediction from "+ v[0]+"s beacon", dpi=2000)
    # plt.show()
    # plt.close()


"""
#############################################################################################
# X = [",".join(item) for item in X_test.astype(str)]
# print("X\n",X)
# c = np.dstack((X,y_pred, std))
# # print(c)
# dataframe = pd.DataFrame(c[0], columns=["location","rssi_mean_GP","rssi_std_GP"])
# display(dataframe)
# b=abs(y_test-y_pred)
# c = np.dstack((y_test,y_pred))
# print(c[0])
# a.append(c[0])
# print(a)

#############################################################################################
# df2.rename(columns={'rssi_mean': 'rssi_mean_GT', 'rssi_std': 'rssi_std_GT' }, inplace=True)
# df_ =pd.merge(df2, dataframe, on='location')
# # df_.set_index("location", inplace = True)
# df_ = df_[['location','rssi_mean_GT','rssi_std_GT','rssi_mean_GP','rssi_std_GP']]
# fig, ax = plt.subplots()
# ax.axis('off')
# ax.axis('tight')
# t= ax.table(cellText=df_.values, colWidths = [0.25]*len(df_.columns), colLabels=df_.columns,  loc='center')
# t.auto_set_font_size(False) 
# t.set_fontsize(8)
# fig.tight_layout()
# plt.title('sit_entry' +" mean_absolute_error :" + str(round(MAE,2)) , y=-0.008)
# plt.savefig('sit_entry' + " comparision table GT vs PL ", dpi=2000)
# plt.show()
###############################################################################################




#################################3d plot#######################################################
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection='3d')            
# surf =ax.plot_surface(X0p, X1p, Zp_mean, rstride=1, cstride=1, cmap='jet', linewidth=0, antialiased=False)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('RSSI')
# # plt.title("Mean of GP prediction from one Beacon(assitech)")
# plt.savefig(v[0]+".pdf", dpi=2000)
# plt.show()


# In[54]:


# dataframe.to_csv('sit_entry_RBF.csv', index=False)



"""
