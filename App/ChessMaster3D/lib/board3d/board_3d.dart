import 'package:flutter/material.dart';
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
    return LayoutBuilder(
      builder: (context, constraints) {
        return Cube(
          onSceneCreated: (s) {
            scene = s;
            s.camera.position.setValues(0, 8, 12);
            s.camera.target.setValues(0, 0, 0);
            board = Object(name: 'board');
            // Simple primitives for fast ship; can swap to OBJ assets later
            s.world.add(board!);
          },
          interactive: true, // orbit + zoom + tilt via gestures
        );
      },
    );
  }
}
