import 'dart:io';

import 'package:chess/chess.dart' as chess;

/// Difficulty levels 1..5
enum Difficulty { beginner, amateur, intermediate, advanced, hard }

extension DifficultyX on Difficulty {
  int get level => index + 1;
  static Difficulty fromLevel(int level) => Difficulty.values[(level - 1).clamp(0, 4)];
}

abstract class AIProvider {
  Future<chess.Move?> bestMove(chess.Chess board, Difficulty difficulty);
}

class StockfishAI implements AIProvider {
  StockfishAI({this.enginePath});
  final String? enginePath;

  @override
  Future<chess.Move?> bestMove(chess.Chess board, Difficulty difficulty) async {
    // Minimal stub: pick a legal move with randomization by difficulty.
    final moves = board.generate_moves();
    if (moves.isEmpty) return null;
    // TODO: wire real Stockfish UCI process and parse bestmove
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
    final moves = board.generate_moves();
    if (moves.isEmpty) return null;
    // Placeholder policy: bias towards captures at higher levels
    moves.sort((a,b){
      final ca = board.get(b.toAlgebraic(a.toString().split(' ').first)) != null ? 1 : 0;
      final cb = board.get(b.toAlgebraic(b.toString().split(' ').first)) != null ? 1 : 0;
      return cb.compareTo(ca);
    });
    return moves.first;
  }
}
