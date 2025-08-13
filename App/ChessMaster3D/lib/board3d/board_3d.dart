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
  bool _loading = true;
  String? _error;

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
    return Stack(children: [
      LayoutBuilder(
        builder: (context, constraints) {
          return Cube(
            onSceneCreated: (s) async {
              scene = s;
              s.camera.position.setValues(0, 6, 6);
              s.camera.target.setValues(0, 0, 0);
              const tiltDeg = 18.0;
              try {
                final boardObj = Object(fileName: 'assets/models/board.obj');
                try {
                  (boardObj as dynamic).backfaceCulling = false;
                } catch (_) {}
                boardObj.rotation.setValues(-tiltDeg, 0, 0);
                board = boardObj;
                s.world.add(boardObj);
                setState(() {
                  _loading = false;
                });
              } catch (e) {
                debugPrint('Board OBJ load error: $e');
                final fallback = Object(name: 'board_fallback');
                fallback.scale.setValues(8, 0.05, 8);
                fallback.rotation.setValues(-tiltDeg, 0, 0);
                board = fallback;
                s.world.add(fallback);
                setState(() {
                  _error = 'Fallback board (OBJ failed)';
                  _loading = false;
                });
              }
            },
            interactive: true, // orbit + zoom + tilt via gestures
          );
        },
      ),
      if (_loading)
        const Positioned.fill(
          child: IgnorePointer(
            child: Center(child: CircularProgressIndicator()),
          ),
        ),
      if (_error != null)
        Positioned(
            left: 8,
            right: 8,
            top: 8,
            child: DecoratedBox(
              decoration: BoxDecoration(
                color: Colors.red.withOpacity(0.1),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Padding(
                padding: const EdgeInsets.all(8.0),
                child: Text(_error!,
                    style: const TextStyle(fontSize: 12, color: Colors.red)),
              ),
            )),
      // Simple help overlay
      Positioned(
        right: 8,
        bottom: 8,
        child: DecoratedBox(
          decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.3),
              borderRadius: BorderRadius.circular(6)),
          child: const Padding(
            padding: EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            child: Text('Drag = orbit\nPinch = zoom',
                style: TextStyle(color: Colors.white, fontSize: 10)),
          ),
        ),
      ),
    ]);
  }
}
