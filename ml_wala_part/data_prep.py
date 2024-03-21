import os
import pandas as pd
import numpy as np

# Define the folder containing CSV files
# to run this python -W ignore .\data_prep.py
floor = "Ground"
folder_path = floor + "/all_gt/"

# Define the fixed set of MAC IDs to filter
# Add your MAC IDs here # Add your MAC IDs here
fixed_mac_ids_third = [
    "7C:D9:F4:01:14:69",
    "7C:D9:F4:03:99:BD",
    "7C:D9:F4:03:AF:F6",
    "7C:D9:F4:09:66:EA",
    "7C:D9:F4:03:4B:CB",
    "7C:D9:F4:07:12:56",
    "7C:D9:F4:01:21:33",
    "7C:D9:F4:09:66:EA",
    "7C:D9:F4:00:B6:FA",
    "7C:D9:F4:01:24:1D",
    "7C:D9:F4:01:B8:83",
    "7C:D9:F4:1D:65:E1",
    "7C:D9:F4:02:8B:FE",
    "7C:D9:F4:03:86:1A",
    "7C:D9:F4:02:4C:FB",
    "7C:D9:F4:02:66:A7",
    "7C:D9:F4:00:41:91",
    "7C:D9:F4:01:13:8F",
]

fixed_mac_ids_ground = [
    "7C:D9:F4:03:84:63",
    "7C:D9:F4:04:02:E4",
    "7C:D9:F4:00:9D:60",
    "7C:D9:F4:02:8B:BA",
    "7C:D9:F4:04:2B:B9",
    "7C:D9:F4:00:45:69",
    "7C:D9:F4:00:C5:40",
    "7C:D9:F4:01:AE:04",
    "7C:D9:F4:02:DE:77",
    "7C:D9:F4:03:A9:A3",
    "7C:D9:F4:00:D5:BD",
    "7C:D9:F4:02:F4:8A",
    "7C:D9:F4:03:C2:6A",
    "7C:D9:F4:19:2C:39",
    "7C:D9:F4:1C:2E:D9",
    "7C:D9:F4:03:A3:3C",
]

if floor == "Ground":
    mac_list = fixed_mac_ids_ground
else:
    mac_list = fixed_mac_ids_third

# Create an empty dictionary to store dataframes for each MAC ID
mac_dataframes = {}
# Loop through each MAC ID
for mac_id in mac_list:
    # Create an empty DataFrame for the current MAC ID
    mac_dataframes[mac_id] = pd.DataFrame()

# Loop through each CSV file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        # Extract the file ID from the filename (assuming filename format is 'gt/{file_id}.csv')
        file_id = filename.split(".")[0]
        # print("File ID",file_id)
        print(file_id)
        # Read the CSV file
        df = pd.read_csv(os.path.join(folder_path, filename))
        df_sample = pd.read_csv(floor + "/data_sample_loc.csv")
        loc_id = df_sample["Coodinate"][int(file_id) - 1]
        print("ID===================", loc_id)
        # Loop through each MAC ID
        for mac_id in fixed_mac_ids_ground:
            # Filter the DataFrame based on the current MAC ID
            filtered_df = df[df["MAC Address"] == mac_id]
            # Add a new column to store the file ID rssi_mean	rssi_std	location
            filtered_df["rssi_mean"] = np.mean(filtered_df["RSSI"])
            filtered_df["rssi_std"] = np.std(filtered_df["RSSI"])
            filtered_df["location"] = loc_id
            # filtered_df['file_id'] = file_id
            print(
                np.mean(filtered_df["RSSI"]),
                np.std(filtered_df["RSSI"]),
                loc_id,
                mac_id,
            )
            # Append the filtered data to the corresponding DataFrame
            mac_dataframes[mac_id] = mac_dataframes[mac_id].append(
                filtered_df, ignore_index=True
            )

for mac in fixed_mac_ids_ground:
    # print(mac_dataframes[mac])
    columns_to_drop = [
        "Sr No",
        "RSSI",
        "Timestamp",
        "Device Name",
        "MAC Address",
        "Raw Data",
        "file_id",
    ]  # specify the columns you want to delete
    columns_exist = [col for col in columns_to_drop if col in df.columns]
    if columns_exist:
        mac_dataframes[mac].drop(columns=columns_exist, inplace=True)

    mac_dataframes[mac] = mac_dataframes[mac].drop_duplicates()
    mac_dataframes[mac].to_csv(
        floor + "/beacons_gt/" + mac.replace(":", "") + ".csv", index=False
    )
