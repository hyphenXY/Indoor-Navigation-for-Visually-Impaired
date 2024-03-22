from flask import Flask, request, jsonify
import csv
import os

app = Flask(__name__)
count = 1  # Initialize index for naming CSV files


@app.route("/")
def index():
    return "Hello, World!"


@app.route("/json", methods=["POST"])
def upload_data():
    global count
    data = request.json  # Assuming data is sent as JSON
    print((data))

    # mac_address = []
    # for i in range(len(data)):
    #     mac_address.append(data[i]["mac"])

    # rssi = []
    # for i in range(len(data)):
    #     rssi.append(data[i]["rssi"])

    # # Create directory if it doesn't exist
    # if not os.path.exists("Project/ground/all_gt"):
    #     os.makedirs("Project/ground/all_gt")

    # # Generate filename with incremented index
    # filename = f"Project/ground/all_gt/{count}.csv"
    # count += 1

    # Write data to CSV file
    # with open(filename, "w", newline="") as csvfile:
    #     fieldnames = ["MAC Address", "RSSI"]
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     for i in range(len(data)):
    #         writer.writerow({"MAC Address": mac_address[i], "RSSI": rssi[i]})

    # run data_prep.py
    # os.system("python Project/data_prep.py")
    # os.system("python Project/rssi_generation.py")
    os.system("python Project/localization.py")

    return jsonify({"message": "Data received and saved successfully"})


if __name__ == "__main__":
    app.run(host="192.168.47.20",debug=True, port=5000)  # Run the app in debug mode

