import 'package:flutter/material.dart';

class AppSettings extends ChangeNotifier {
  ThemeMode _themeMode = ThemeMode.system;
  bool _use3D = false;

  ThemeMode get themeMode => _themeMode;
  bool get use3D => _use3D;

  void setThemeMode(ThemeMode mode) {
    if (_themeMode == mode) return;
    _themeMode = mode;
    notifyListeners();
  }

  void setUse3D(bool value) {
    if (_use3D == value) return;
    _use3D = value;
    notifyListeners();
  }
}
