from flask import Flask, request, jsonify
import csv
import os

app = Flask(__name__)
count = 0  # Initialize index for naming CSV files


@app.route("/")
def index():
    return "Hello, World!"


@app.route("/json", methods=["POST"])
def upload_data():
    global count
    data = request.json  # Assuming data is sent as JSON
    print((data))
    # convert data to list
    # get mac from json
    # get rssi from json

    mac_address = []
    for i in range(len(data)):
        mac_address.append(data[i]["mac"])

    rssi = []
    for i in range(len(data)):
        rssi.append(data[i]["rssi"])

    # Create directory if it doesn't exist
    if not os.path.exists("Project/ground/all_gt"):
        os.makedirs("Project/ground/all_gt")

    # Generate filename with incremented index
    filename = f"Project/ground/all_gt/data_{count}.csv"
    count += 1

    # Write data to CSV file
    with open(filename, "w", newline="") as csvfile:
        fieldnames = ["MAC Address", "RSSI"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"MAC Address": mac_address, "RSSI": rssi})

    return jsonify({"message": "Data received and saved successfully"})


if __name__ == "__main__":
    app.run(host="192.168.47.20",debug=True, port=5000)  # Run the app in debug mode
