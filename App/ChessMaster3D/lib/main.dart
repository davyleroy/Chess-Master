import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'settings/app_settings.dart';
import 'ui/home.dart';

void main() {
  runApp(const ChessMasterApp());
}

class ChessMasterApp extends StatelessWidget {
  const ChessMasterApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => AppSettings(),
      child: Consumer<AppSettings>(
        builder: (context, settings, _) => MaterialApp(
          title: 'Chess Master',
          theme: ThemeData(
              colorSchemeSeed: Colors.indigo, brightness: Brightness.light),
          darkTheme: ThemeData(
              colorSchemeSeed: Colors.indigo, brightness: Brightness.dark),
          themeMode: settings.themeMode,
          home: const HomeScreen(),
        ),
      ),
    );
  }
}
