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
      GameState(
          game: game ?? this.game,
          mode: mode ?? this.mode,
          whiteOnTop: whiteOnTop ?? this.whiteOnTop);
}

class MoveUtils {
  static List<String> legalMovesSAN(chess.Chess g) =>
      g.generate_moves().map((m) => g.move_to_san(m)).toList();
  static String randomMoveSAN(chess.Chess g) {
    final ms = g.generate_moves();
    if (ms.isEmpty) return '';
    ms.shuffle(Random());
    return g.move_to_san(ms.first);
  }
}

class GameController {
  GameController({this.mode = GameMode.single}) : _game = chess.Chess();

  final GameMode mode;
  final chess.Chess _game;

  chess.Chess get game => _game;

  void newGame() {
    _game.reset();
  }

  List<String> legalTargets(String from) {
    return _game
        .moves({'square': from, 'verbose': true})
        .map<String>((m) => m['to'] as String)
        .toList();
  }

  bool move(String from, String to) {
    // Validate first to avoid relying on return types
    final legal = legalTargets(from);
    if (!legal.contains(to)) return false;
    final mv = {'from': from, 'to': to};
    _game.move(mv);
    return true;
  }

  bool get gameOver => _game.game_over;

  bool moveRaw(chess.Move mv) {
    final res = _game.move(mv);
    return res;
  }
}
