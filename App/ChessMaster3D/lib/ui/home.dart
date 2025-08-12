import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../board3d/board_3d.dart';
import '../board2d/board_2d.dart';
import '../engine/game.dart';
import '../settings/app_settings.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool use3D = false; // default to 2D for broader compatibility (web/desktop)
  GameMode mode = GameMode.single;
  int difficulty = 3; // 1..5
  String windowPreset = 'Auto';

  @override
  void initState() {
    super.initState();
    _loadPrefs();
  }

  Future<void> _loadPrefs() async {
    final sp = await SharedPreferences.getInstance();
    setState(() {
      use3D = sp.getBool('use3D') ?? true;
      difficulty = sp.getInt('difficulty') ?? 3;
      windowPreset = sp.getString('windowPreset') ?? 'Auto';
    });
  }

  Future<void> _savePrefs() async {
    final sp = await SharedPreferences.getInstance();
    await sp.setBool('use3D', use3D);
    await sp.setInt('difficulty', difficulty);
    await sp.setString('windowPreset', windowPreset);
  }

  @override
  Widget build(BuildContext context) {
    final appSettings = context.watch<AppSettings>();
    final effectiveUse3D = !kIsWeb && use3D; // fallback to 2D on Web
    return Scaffold(
      appBar: AppBar(title: const Text('Chess Master')),
      drawer: Drawer(
        child: ListView(
          children: [
            const DrawerHeader(child: Text('Chess Master')),
            SwitchListTile(
              title: const Text('Use 3D Board'),
              value: use3D,
              onChanged: (v) async {
                setState(() => use3D = v);
                await _savePrefs();
              },
            ),
            if (kIsWeb && use3D)
              const Padding(
                padding: EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                child: Text(
                    '3D is not available on Web with this plugin. Auto-falling back to 2D.',
                    style: TextStyle(fontSize: 12)),
              ),
            const Divider(),
            ListTile(
              title: const Text('Theme: System'),
              leading: Radio<ThemeMode>(
                value: ThemeMode.system,
                groupValue: appSettings.themeMode,
                onChanged: (m) => appSettings.setThemeMode(m!),
              ),
              onTap: () => appSettings.setThemeMode(ThemeMode.system),
            ),
            ListTile(
              title: const Text('Theme: Light'),
              leading: Radio<ThemeMode>(
                value: ThemeMode.light,
                groupValue: appSettings.themeMode,
                onChanged: (m) => appSettings.setThemeMode(m!),
              ),
              onTap: () => appSettings.setThemeMode(ThemeMode.light),
            ),
            ListTile(
              title: const Text('Theme: Dark'),
              leading: Radio<ThemeMode>(
                value: ThemeMode.dark,
                groupValue: appSettings.themeMode,
                onChanged: (m) => appSettings.setThemeMode(m!),
              ),
              onTap: () => appSettings.setThemeMode(ThemeMode.dark),
            ),
            ListTile(
              title: const Text('Single Player'),
              onTap: () {
                setState(() => mode = GameMode.single);
                Navigator.pop(context);
              },
            ),
            ListTile(
              title: const Text('Local Two Players'),
              onTap: () {
                setState(() => mode = GameMode.local);
                Navigator.pop(context);
              },
            ),
            ListTile(
              title: const Text('Online (coming soon)'),
              onTap: () {
                showDialog(
                    context: context,
                    builder: (_) => const AlertDialog(
                          title: Text('Online Mode'),
                          content: Text(
                              'To be integrated. Falling back to offline modes.'),
                        ));
              },
            ),
            const Divider(),
            ListTile(
              title: const Text('Difficulty'),
              subtitle: Text('$difficulty'),
              onTap: () async {
                final v = await showDialog<int>(
                    context: context,
                    builder: (_) => _DifficultyDialog(current: difficulty));
                if (v != null) {
                  setState(() => difficulty = v);
                  await _savePrefs();
                }
              },
            ),
            ListTile(
              title: const Text('Window Preset'),
              subtitle: Text(windowPreset),
              onTap: () async {
                final v = await showDialog<String>(
                    context: context,
                    builder: (_) => _WindowDialog(current: windowPreset));
                if (v != null) {
                  setState(() => windowPreset = v);
                  await _savePrefs();
                }
              },
            ),
          ],
        ),
      ),
      body: effectiveUse3D
          ? const Board3D()
          : Board2D(mode: mode, difficulty: difficulty),
    );
  }
}

class _DifficultyDialog extends StatefulWidget {
  const _DifficultyDialog({required this.current});
  final int current;
  @override
  State<_DifficultyDialog> createState() => _DifficultyDialogState();
}

class _DifficultyDialogState extends State<_DifficultyDialog> {
  late int v = widget.current;
  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Select Difficulty'),
      content: Column(mainAxisSize: MainAxisSize.min, children: [
        for (final i in [1, 2, 3, 4, 5])
          RadioListTile<int>(
            title: Text({
              1: 'Beginner',
              2: 'Amateur',
              3: 'Intermediate',
              4: 'Advanced',
              5: 'Hard'
            }[i]!),
            value: i,
            groupValue: v,
            onChanged: (x) {
              setState(() => v = x!);
            },
          ),
      ]),
      actions: [
        TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel')),
        FilledButton(
            onPressed: () => Navigator.pop(context, v),
            child: const Text('OK')),
      ],
    );
  }
}

class _WindowDialog extends StatefulWidget {
  const _WindowDialog({required this.current});
  final String current;
  @override
  State<_WindowDialog> createState() => _WindowDialogState();
}

class _WindowDialogState extends State<_WindowDialog> {
  late String v = widget.current;
  final options = const ['Auto', 'Square', '16:9', '4:3'];
  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Window Preset'),
      content: Column(mainAxisSize: MainAxisSize.min, children: [
        for (final o in options)
          RadioListTile<String>(
            title: Text(o),
            value: o,
            groupValue: v,
            onChanged: (x) {
              setState(() => v = x!);
            },
          )
      ]),
      actions: [
        TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel')),
        FilledButton(
            onPressed: () => Navigator.pop(context, v),
            child: const Text('OK')),
      ],
    );
  }
}
