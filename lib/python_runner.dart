import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

void sendListToServer(List<String> deviceNames) async {
  // Read JSON file
  File jsonFile = File('data.json');
  if (!jsonFile.existsSync()) {
    print('Error: JSON file not found.');
    return;
  }

  // Read JSON content
  String jsonString = jsonFile.readAsStringSync();
  Map<String, dynamic> jsonData = json.decode(jsonString);

  // Send JSON data to Flask server
  Uri url = Uri.parse('http://192.168.47.20:5000/json'); // Replace with your server URL
  try {
    http.Response response = await http.post(
      url,
      headers: {'Content-Type': 'application/json'},
      body: json.encode(jsonData),
    );

    // Check response status
    if (response.statusCode == 200) {
      print('JSON data sent successfully.');
    } else {
      print('Failed to send JSON data. Status code: ${response.statusCode}');
    }
  } catch (e) {
    print('Error sending JSON data: $e');
  }
}
