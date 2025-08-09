import 'dart:math';
import 'package:chess/chess.dart' as chess;

enum GameMode { single, local, online }

class GameState {
  final chess.Chess game;
  final GameMode mode;
  final bool whiteOnTop;

  GameState({required this.game, required this.mode, this.whiteOnTop = true});

  bool get gameOver => game.game_over;
  bool get whiteToMove => game.turn == chess.Color.WHITE;

  GameState copyWith({chess.Chess? game, GameMode? mode, bool? whiteOnTop}) =>
      GameState(game: game ?? this.game, mode: mode ?? this.mode, whiteOnTop: whiteOnTop ?? this.whiteOnTop);
}

class MoveUtils {
  static List<String> legalMovesSAN(chess.Chess g) => g.generate_moves().map((m) => g.move_to_san(m)).toList();
  static String randomMoveSAN(chess.Chess g) {
    final ms = g.generate_moves();
    if (ms.isEmpty) return '';
    ms.shuffle(Random());
    return g.move_to_san(ms.first);
  }
}
