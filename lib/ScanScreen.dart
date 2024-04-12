import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:http/http.dart' as http;

import 'ScanResultTile.dart';
import 'utils/snackbar.dart';

class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  final List<BluetoothDevice> _systemDevices = [];
  List<ScanResult> _scanResults = [];
  bool _isScanning = false;
  late StreamSubscription<List<ScanResult>> _scanResultsSubscription;
  late StreamSubscription<bool> _isScanningSubscription;

  @override
  void initState() {
    super.initState();

    _scanResultsSubscription =
        FlutterBluePlus.scanResults.listen((results) {
          _scanResults = results;
          if (mounted) {
            setState(() {});
          }
        }, onError: (e) {
          Snackbar.show(ABC.b, prettyException("Scan Error:", e), success: false);
        });

    _isScanningSubscription =
        FlutterBluePlus.isScanning.listen((state) {
          _isScanning = state;
          if (mounted) {
            setState(() {});
          }
        });
  }

  @override
  void dispose() {
    _scanResultsSubscription.cancel();
    _isScanningSubscription.cancel();
    super.dispose();
  }

  void startScan() {
    if (!_isScanning) {
      try {
        FlutterBluePlus.startScan(timeout: null);
      } catch (e) {
        Snackbar.show(ABC.b, prettyException("Start Scan Error:", e),
            success: false);
      }
      setState(() {
        _isScanning = true;
      });
    }
  }

  void stopScan() {
    if (_isScanning) {
      try {
        FlutterBluePlus.stopScan();
      } catch (e) {
        Snackbar.show(ABC.b, prettyException("Stop Scan Error:", e),
            success: false);
      }
      setState(() {
        _isScanning = false;
      });
    }
  }

  void sendDataToServer() async {
    if (_scanResults.isNotEmpty) {
      await _sendDataToServer();
    }
  }

  Future<void> refreshAndStartScan() async {
    startScan();
  }


  @override
  Widget build(BuildContext context) {
    return ScaffoldMessenger(
      key: Snackbar.snackBarKeyB,
      child: Scaffold(
        appBar: AppBar(
          title: const Text('Find Devices'),
          backgroundColor: Colors.purple.shade300,
        ),
        body: RefreshIndicator(
          onRefresh: refreshAndStartScan,
          child: ListView(
            children: <Widget>[
              ..._buildScanResultTiles(context),
            ],
          ),
        ),
        floatingActionButton: Column(
          mainAxisAlignment: MainAxisAlignment.end,
          crossAxisAlignment: CrossAxisAlignment.end,
          children: [
            FloatingActionButton(
              onPressed: startScan,
              backgroundColor: Colors.green.shade400,
              tooltip: 'Scan Start',
              child: const Icon(Icons.bluetooth_searching),
            ),
            const SizedBox(height: 10),
            FloatingActionButton(
              onPressed: stopScan,
              backgroundColor: Colors.red.shade500,
              tooltip: 'Scan Stop',
              child: const Icon(Icons.stop),
            ),
            const SizedBox(height: 10),
            FloatingActionButton(
              onPressed: sendDataToServer,
              backgroundColor: Colors.amber.shade300,
              tooltip: 'Send to Server',
              child: const Icon(Icons.send),
            ),
          ],
        ),
      ),
    );
  }

  List<Widget> _buildScanResultTiles(BuildContext context) {
    return _scanResults
        .map(
          (r) => ScanResultTile(
        result: r,
      ),
    )
        .toList();
  }

  Future<void> _sendDataToServer() async {
    List<Map<String, dynamic>> devicesData = [];

    for (var result in _scanResults) {
      devicesData.add({
        'mac': result.device.remoteId.str,
        'rssi': result.rssi.toString(),
      });
    }

    final jsonData = jsonEncode(devicesData);

    try {
      final response = await http.post(
        Uri.parse('http://192.168.47.20:5000/json'),
        headers: <String, String>{
          'Content-Type': 'application/json; charset=UTF-8',
        },
        body: jsonData,
      );

      if (response.statusCode == 200) {
        if (kDebugMode) {
          print('Data sent successfully');
        }
      } else {
        throw Exception('Failed to send data: ${response.statusCode}');
      }
    } catch (e) {
      if (kDebugMode) {
        print('Error sending data: $e');
      }
      Snackbar.show(ABC.b, prettyException("Send Data Error:", e),
          success: false);
    }
  }
}
