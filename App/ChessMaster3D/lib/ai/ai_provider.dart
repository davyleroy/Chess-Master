import 'package:chess/chess.dart' as chess;

/// Difficulty levels 1..5
enum Difficulty { beginner, amateur, intermediate, advanced, hard }

extension DifficultyX on Difficulty {
  int get level => index + 1;
  static Difficulty fromLevel(int level) =>
      Difficulty.values[(level - 1).clamp(0, 4)];
}

abstract class AIProvider {
  Future<chess.Move?> bestMove(chess.Chess board, Difficulty difficulty);
}

class StockfishAI implements AIProvider {
  StockfishAI({this.enginePath});
  final String? enginePath;

  @override
  Future<chess.Move?> bestMove(chess.Chess board, Difficulty difficulty) async {
    // Simple heuristic until real Stockfish is wired:
    // Prefer # (mate) > captures (x) > checks (+) > others. Random within tier.
    final moves = board.generate_moves();
    if (moves.isEmpty) return null;

    List<chess.Move> tier(String pattern) =>
        moves.where((m) => board.move_to_san(m).contains(pattern)).toList();
    final mates = tier('#');
    if (mates.isNotEmpty) {
      mates.shuffle();
      return mates.first;
    }
    final captures = tier('x');
    if (captures.isNotEmpty) {
      // Higher difficulty: bias towards capturing higher-value pieces by SAN letter order (Q>R>B/N>P)
      if (difficulty.level >= 4) {
        captures.sort((a, b) {
          final sa = board.move_to_san(a);
          final sb = board.move_to_san(b);
          int score(String s) {
            // crude: prioritize captures with piece letters in SAN target; fallback 0
            if (s.contains('Q')) return 9;
            if (s.contains('R')) return 5;
            if (s.contains('B') || s.contains('N')) return 3;
            if (s.contains('x')) return 1;
            return 0;
          }

          return score(sb).compareTo(score(sa));
        });
        return captures.first;
      }
      captures.shuffle();
      return captures.first;
    }
    final checks = tier('+');
    if (checks.isNotEmpty && difficulty.level >= 3) {
      checks.shuffle();
      return checks.first;
    }
    // Otherwise, random move; slightly center-bias at high difficulty by SAN with e/d files
    moves.shuffle();
    return moves.first;
  }
}

class NeuralAI implements AIProvider {
  NeuralAI({required this.modelAssetPath});
  final String modelAssetPath; // e.g., assets/models/chess_model.pt

  bool _loaded = false;

  Future<void> _ensureLoaded() async {
    if (_loaded) return;
    // TODO: load TorchScript via mobile plugin; desktop: noop
    _loaded = true;
  }

  @override
  Future<chess.Move?> bestMove(chess.Chess board, Difficulty difficulty) async {
    await _ensureLoaded();
    // Mirror the same simple heuristic as above for now
    final moves = board.generate_moves();
    if (moves.isEmpty) return null;

    List<chess.Move> tier(String pattern) =>
        moves.where((m) => board.move_to_san(m).contains(pattern)).toList();
    final mates = tier('#');
    if (mates.isNotEmpty) {
      mates.shuffle();
      return mates.first;
    }
    final captures = tier('x');
    if (captures.isNotEmpty) {
      if (difficulty.level >= 4) {
        captures.sort((a, b) {
          final sa = board.move_to_san(a);
          final sb = board.move_to_san(b);
          int score(String s) {
            if (s.contains('Q')) return 9;
            if (s.contains('R')) return 5;
            if (s.contains('B') || s.contains('N')) return 3;
            if (s.contains('x')) return 1;
            return 0;
          }

          return score(sb).compareTo(score(sa));
        });
        return captures.first;
      }
      captures.shuffle();
      return captures.first;
    }
    final checks = tier('+');
    if (checks.isNotEmpty && difficulty.level >= 3) {
      checks.shuffle();
      return checks.first;
    }
    moves.shuffle();
    return moves.first;
  }
}
