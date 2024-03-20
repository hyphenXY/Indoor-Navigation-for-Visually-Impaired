import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

void sendListToServer(List<dynamic> myList) async {
  // Encode your list to JSON
  String jsonList = jsonEncode(myList);

  // Define the URL of your Flask server endpoint
  String url = 'http://localhost:5000/';

  try {
    // Make a POST request to send the list
    final response = await http.post(
      Uri.parse(url),
      headers: <String, String>{
        'Content-Type': 'application/json; charset=UTF-8',
      },
      body: jsonList,
    );

    // Check if the request was successful
    if (response.statusCode == 200) {
      if (kDebugMode) {
          print('List sent successfully');
      }
    } else {
      if (kDebugMode) {
        print('Failed to send list. Status code: ${response.statusCode}');
      }
    }
  } catch (e) {
    if (kDebugMode) {
      print('Error sending list: $e');
    }
  }
}
