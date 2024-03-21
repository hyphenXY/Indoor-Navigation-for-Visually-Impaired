from flask import Flask, request

app = Flask(__name__)


@app.route("/")
def index():
    return "Hello, World!"


@app.route("/json", methods=["POST"])
def android_data():
    data = request.json  # Assuming data is sent as JSON
    print(data)
    return "Data received"


if __name__ == "__main__":
    app.run(host="192.168.47.20", debug=True, port=5000)
