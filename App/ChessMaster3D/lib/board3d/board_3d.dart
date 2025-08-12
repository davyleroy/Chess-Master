import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter_cube/flutter_cube.dart';

class Board3D extends StatefulWidget {
  const Board3D({super.key});

  @override
  State<Board3D> createState() => _Board3DState();
}

class _Board3DState extends State<Board3D> {
  Scene? scene;
  Object? board;

  @override
  void initState() {
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    if (kIsWeb) {
      return const Center(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Text(
              '3D board is not available on Web with this plugin. Switch to 2D or run on Windows/Android/iOS.'),
        ),
      );
    }
    return LayoutBuilder(
      builder: (context, constraints) {
        return Cube(
          onSceneCreated: (s) {
            scene = s;
            s.camera.position.setValues(6, 8, 10); // tilted
            s.camera.target.setValues(0, 0, 0);
            // flutter_cube has ambient light by default; no explicit light arg.
            try {
              board = Object(fileName: 'assets/models/board.obj');
              board!.rotation.setValues(-90, 0, 0); // lay flat if needed
              s.world.add(board!);
            } catch (_) {
              board = Object(name: 'board');
              s.world.add(board!);
            }
          },
          interactive: true, // orbit + zoom + tilt via gestures
        );
      },
    );
  }
}
