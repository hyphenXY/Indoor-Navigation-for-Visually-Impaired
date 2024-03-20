import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

//write code to send a list which is inputed from main.dart to the server and get the response from the server and display it in the main.dart
//i already have main.dart file

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final TextEditingController _controller = TextEditingController();
  final List<String> _items = [];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('HTTP POST request and response'),
      ),
      body: Column(
        children: <Widget>[
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: TextField(
              controller: _controller,
              decoration: const InputDecoration(
                labelText: 'Enter a message',
              ),
            ),
          ),
          Expanded(
            child: ListView.builder(
              itemCount: _items.length,
              itemBuilder: (BuildContext context, int index) {
                return ListTile(
                  title: Text(_items[index]),
                );
              },
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          final String message = _controller.text;
          final String url = 'https://jsonplaceholder.typicode.com/posts';
          final Response response = await post(
            Uri.parse(url),
            headers: <String, String>{
              'Content-Type': 'application/json; charset=UTF-8',
            },
            body: jsonEncode(<String, String>{
              'title': message,
            }),
          );
          if (response.statusCode == 201) {
            final Map<String, dynamic> data = jsonDecode(response.body);
            setState(() {
              _items.add(data['title']);
            });
          } else {
            throw Exception('Failed to load data');
          }
        },
        tooltip: 'Send message',
        child: const Icon(Icons.send),
      ),
    );
  }
}