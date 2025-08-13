import 'package:flutter/material.dart';
import 'package:chess/chess.dart' as chess;
import '../ai/ai_provider.dart';
import '../engine/game.dart';

class _StartOptions {
  final GameMode mode;
  final int difficulty;
  final chess.Color humanColor; // for single-player
  final _TimeControl? timeControl; // null => untimed
  const _StartOptions(
      this.mode, this.difficulty, this.humanColor, this.timeControl);
}

class _TimeControl {
  final Duration base;
  final Duration increment;
  const _TimeControl(this.base, this.increment);
  String label() {
    if (increment.inSeconds == 0) {
      return "${base.inMinutes}:${(base.inSeconds % 60).toString().padLeft(2, '0')}";
    }
    return "${base.inMinutes}+${increment.inSeconds}";
  }
}

class Board2D extends StatefulWidget {
  const Board2D({super.key, this.mode = GameMode.single, this.difficulty = 3});
  final GameMode mode;
  final int difficulty; // 1..5

  @override
  State<Board2D> createState() => _Board2DState();
}

class _Board2DState extends State<Board2D> {
  late GameController ctrl;
  String? selected;
  List<String> targets = const [];
  late GameMode _mode;
  late int _difficulty;
  late AIProvider _ai;
  bool _aiThinking = false;
  final List<String> _capturedWhite =
      []; // pieces captured from White (shown near Black)
  final List<String> _capturedBlack =
      []; // pieces captured from Black (shown near White)
  chess.Color _humanColor = chess.Color.WHITE; // default human is White
  _TimeControl? _timeControl;
  Duration? _timeWhite;
  Duration? _timeBlack;
  chess.Color _sideToMoveClock = chess.Color.WHITE;
  bool _flagged = false;
  chess.Color? _winnerOnTime;
  DateTime? _lastTick;

  @override
  void initState() {
    super.initState();
    _mode = widget.mode;
    _difficulty = widget.difficulty;
    ctrl = GameController(mode: _mode);
    _ai = StockfishAI();
    _startTickLoop();
  }

  void _startTickLoop() {
    _lastTick = DateTime.now();
    Future.doWhile(() async {
      await Future.delayed(const Duration(milliseconds: 250));
      if (!mounted) return false;
      final now = DateTime.now();
      if (_timeControl != null && !_flagged && !ctrl.gameOver) {
        final dt = now.difference(_lastTick!);
        if (dt > Duration.zero) {
          setState(() {
            if (_sideToMoveClock == chess.Color.WHITE) {
              _timeWhite = (_timeWhite! - dt);
              if (_timeWhite!.inMilliseconds <= 0) {
                _timeWhite = Duration.zero;
                _flagged = true;
                _winnerOnTime = chess.Color.BLACK;
              }
            } else {
              _timeBlack = (_timeBlack! - dt);
              if (_timeBlack!.inMilliseconds <= 0) {
                _timeBlack = Duration.zero;
                _flagged = true;
                _winnerOnTime = chess.Color.WHITE;
              }
            }
          });
        }
      }
      _lastTick = now;
      return true; // continue loop
    });
  }

  void _resetClock(_TimeControl? tc) {
    _timeControl = tc;
    if (tc == null) {
      _timeWhite = null;
      _timeBlack = null;
      _flagged = false;
      _winnerOnTime = null;
      return;
    }
    _timeWhite = tc.base;
    _timeBlack = tc.base;
    _flagged = false;
    _winnerOnTime = null;
    _sideToMoveClock = chess.Color.WHITE;
  }

  @override
  Widget build(BuildContext context) {
    // Use Expanded + AspectRatio to avoid vertical overflow on small screens.
    final squares = _buildSquares();
    final over = ctrl.gameOver;
    final result = over ? _resultText() : null;
    return SafeArea(
      child: Column(
        children: [
          // Captured by White (i.e., black pieces taken)
          _capturedRow(_capturedBlack, alignEnd: false),
          if (_timeControl != null) _clocksRow(),
          Expanded(
            child: Center(
              child: AspectRatio(
                aspectRatio: 1,
                child: GridView.builder(
                  padding: EdgeInsets.zero,
                  physics: const NeverScrollableScrollPhysics(),
                  itemCount: 64,
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                      crossAxisCount: 8),
                  itemBuilder: (_, i) => squares[i],
                ),
              ),
            ),
          ),
          // Captured by Black (i.e., white pieces taken)
          _capturedRow(_capturedWhite, alignEnd: true),
          const SizedBox(height: 8),
          Wrap(spacing: 8, children: [
            FilledButton(
              onPressed: () async {
                final opts = await _showNewGameDialog();
                if (opts == null) return;
                setState(() {
                  _mode = opts.mode;
                  _difficulty = opts.difficulty;
                  _humanColor = opts.humanColor;
                  _resetClock(opts.timeControl);
                  ctrl = GameController(mode: _mode);
                  selected = null;
                  targets = const [];
                  _capturedWhite.clear();
                  _capturedBlack.clear();
                });
                // If later you want AI to play White, trigger here when turn is White.
                _maybeAiTurn();
              },
              child: const Text('Start / New Game'),
            ),
          ]),
          if (result != null)
            Padding(
              padding: const EdgeInsets.symmetric(vertical: 8.0),
              child: Text(
                result,
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
            ),
          const SizedBox(height: 12),
        ],
      ),
    );
  }

  List<Widget> _buildSquares() {
    final g = ctrl.game;
    final List<Widget> out = [];
    // Ranks 8..1 (top to bottom), files a..h (left to right)
    for (int rank = 7; rank >= 0; rank--) {
      for (int file = 0; file < 8; file++) {
        // index not needed; compute square directly
        final fileChar = String.fromCharCode('a'.codeUnitAt(0) + file);
        final sq = '$fileChar${rank + 1}';
        final piece = g.get(sq);
        final dark = (file + rank) % 2 == 1;
        final isSel = selected == sq;
        final isTarget = targets.contains(sq);
        out.add(GestureDetector(
          onTap: () => _onSquareTap(sq, piece != null),
          child: Container(
            decoration: BoxDecoration(
              color: isSel
                  ? Colors.teal.withValues(alpha: 0.6)
                  : isTarget
                      ? Colors.orange.withValues(alpha: 0.5)
                      : (dark ? Colors.brown[700] : Colors.brown[200]),
              border: Border.all(color: Colors.black12, width: 0.5),
            ),
            child: Center(child: _pieceGlyph(piece)),
          ),
        ));
      }
    }
    return out;
  }

  void _onSquareTap(String sq, bool hasPiece) {
    // Block input on AI's turn in single-player
    if (_mode == GameMode.single && ctrl.game.turn != _humanColor) {
      return;
    }
    if (selected == null) {
      if (!hasPiece) return;
      // Only allow selecting the side to move
      final p = ctrl.game.get(sq);
      if (p == null || p.color != ctrl.game.turn) return;
      if (_mode == GameMode.single && p.color != _humanColor) return;
      setState(() {
        selected = sq;
        targets = ctrl.legalTargets(sq);
      });
      return;
    }
    // If tapping another of your pieces, reselect.
    final piece = ctrl.game.get(sq);
    if (piece != null && targets.contains(sq) == false) {
      setState(() {
        selected = sq;
        targets = ctrl.legalTargets(sq);
      });
      return;
    }
    // Attempt move from selected to sq
    final from = selected!;
    // Snapshot before move for capture detection
    final mover = ctrl.game.turn; // color to move before applying
    final before = _snapshotBoard();
    final ok = ctrl.move(from, sq);
    setState(() {
      if (ok) {
        final after = _snapshotBoard();
        _recordCaptureByDiff(before, after, mover);
        _applyIncrementAndSwitch(mover);
        selected = null;
        targets = const [];
      } else {
        // Keep selection if illegal
        selected = from;
        targets = ctrl.legalTargets(from);
      }
    });
    if (ok && ctrl.gameOver) _showGameOverDialog();
    // After player's successful move, let AI respond in single-player
    if (ok) _maybeAiTurn();
  }

  Widget _pieceGlyph(chess.Piece? p) {
    if (p == null) return const SizedBox.shrink();
    final isWhite = p.color == chess.Color.WHITE;
    final color = isWhite ? Colors.white : Colors.black87;
    final code = switch (p.type) {
      chess.PieceType.PAWN => '♙',
      chess.PieceType.ROOK => '♖',
      chess.PieceType.KNIGHT => '♘',
      chess.PieceType.BISHOP => '♗',
      chess.PieceType.QUEEN => '♕',
      chess.PieceType.KING => '♔',
      _ => '?',
    };
    // Use black set for black pieces
    final text = isWhite
        ? code
        : {'♙': '♟', '♖': '♜', '♘': '♞', '♗': '♝', '♕': '♛', '♔': '♚'}[code] ??
            code;
    return Text(text, style: TextStyle(fontSize: 28, color: color));
  }

  Future<_StartOptions?> _showNewGameDialog() async {
    GameMode selMode = _mode;
    int selDiff = _difficulty;
    chess.Color selColor = _humanColor;
    _TimeControl? selTc = _timeControl;
    final presets = <_TimeControl?>[
      null,
      const _TimeControl(Duration(minutes: 1), Duration.zero),
      const _TimeControl(Duration(minutes: 3), Duration(seconds: 2)),
      const _TimeControl(Duration(minutes: 5), Duration.zero),
      const _TimeControl(Duration(minutes: 10), Duration.zero),
      const _TimeControl(Duration(minutes: 15), Duration(seconds: 10)),
      const _TimeControl(Duration(minutes: 30), Duration.zero),
    ];
    return showDialog<_StartOptions>(
      context: context,
      builder: (_) => StatefulBuilder(
        builder: (ctx, setL) => AlertDialog(
          title: const Text('Start Game'),
          content: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Mode'),
                const SizedBox(height: 4),
                RadioListTile<GameMode>(
                  title: const Text('Single Player'),
                  value: GameMode.single,
                  groupValue: selMode,
                  onChanged: (v) => setL(() => selMode = v!),
                ),
                RadioListTile<GameMode>(
                  title: const Text('Two Players (Local)'),
                  value: GameMode.local,
                  groupValue: selMode,
                  onChanged: (v) => setL(() => selMode = v!),
                ),
                const Divider(),
                if (selMode == GameMode.single) ...[
                  const Text('Human plays'),
                  RadioListTile<chess.Color>(
                    title: const Text('White'),
                    value: chess.Color.WHITE,
                    groupValue: selColor,
                    onChanged: (v) => setL(() => selColor = v!),
                  ),
                  RadioListTile<chess.Color>(
                    title: const Text('Black'),
                    value: chess.Color.BLACK,
                    groupValue: selColor,
                    onChanged: (v) => setL(() => selColor = v!),
                  ),
                  const Divider(),
                ],
                const Text('Difficulty'),
                for (final i in [1, 2, 3, 4, 5])
                  RadioListTile<int>(
                    title: Text({
                      1: 'Beginner',
                      2: 'Amateur',
                      3: 'Intermediate',
                      4: 'Advanced',
                      5: 'Hard',
                    }[i]!),
                    value: i,
                    groupValue: selDiff,
                    onChanged: (x) => setL(() => selDiff = x!),
                  ),
                const Divider(),
                const Text('Time Control'),
                for (final tc in presets)
                  RadioListTile<_TimeControl?>(
                    title: Text(tc == null ? 'Untimed' : tc.label()),
                    value: tc,
                    groupValue: selTc,
                    onChanged: (v) => setL(() => selTc = v),
                  ),
              ],
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
            FilledButton(
              onPressed: () => Navigator.pop(
                  context,
                  _StartOptions(
                      selMode,
                      selDiff,
                      selMode == GameMode.single ? selColor : chess.Color.WHITE,
                      selTc)),
              child: const Text('Start'),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _maybeAiTurn() async {
    if (_mode != GameMode.single) return;
    if (ctrl.gameOver) return;
    final aiColor = _humanColor == chess.Color.WHITE
        ? chess.Color.BLACK
        : chess.Color.WHITE;
    // AI moves only on its color's turn
    if (ctrl.game.turn != aiColor) return;
    if (_aiThinking) return;
    _aiThinking = true;
    try {
      final move =
          await _ai.bestMove(ctrl.game, DifficultyX.fromLevel(_difficulty));
      await Future.delayed(const Duration(seconds: 2));
      if (move != null) {
        setState(() {
          final mover = ctrl.game.turn; // AI mover color
          final before = _snapshotBoard();
          ctrl.moveRaw(move);
          final after = _snapshotBoard();
          _recordCaptureByDiff(before, after, mover);
          _applyIncrementAndSwitch(mover);
        });
        if (ctrl.gameOver) _showGameOverDialog();
      }
    } finally {
      _aiThinking = false;
    }
  }

  void _applyIncrementAndSwitch(chess.Color mover) {
    if (_timeControl == null) return;
    final inc = _timeControl!.increment;
    if (_timeWhite == null || _timeBlack == null) return;
    if (mover == chess.Color.WHITE) {
      _timeWhite = _timeWhite! + inc;
      _sideToMoveClock = chess.Color.BLACK;
    } else {
      _timeBlack = _timeBlack! + inc;
      _sideToMoveClock = chess.Color.WHITE;
    }
  }

  Map<String, chess.Piece?> _snapshotBoard() {
    final Map<String, chess.Piece?> m = {};
    for (int rank = 1; rank <= 8; rank++) {
      for (int file = 0; file < 8; file++) {
        final fileChar = String.fromCharCode('a'.codeUnitAt(0) + file);
        final sq = '$fileChar$rank';
        m[sq] = ctrl.game.get(sq);
      }
    }
    return m;
  }

  void _recordCaptureByDiff(Map<String, chess.Piece?> before,
      Map<String, chess.Piece?> after, chess.Color mover) {
    chess.Piece? captured;
    for (int rank = 1; rank <= 8; rank++) {
      for (int file = 0; file < 8; file++) {
        final fileChar = String.fromCharCode('a'.codeUnitAt(0) + file);
        final sq = '$fileChar$rank';
        final b = before[sq];
        final a = after[sq];
        if (b != null && b.color != mover) {
          if (a == null || a.color == mover) {
            captured = b;
            break;
          }
        }
      }
      if (captured != null) break;
    }
    if (captured != null) {
      final glyph = _glyphFromType(_typeCode(captured.type), captured.color);
      if (mover == chess.Color.WHITE) {
        _capturedBlack.add(glyph);
      } else {
        _capturedWhite.add(glyph);
      }
    }
  }

  String _typeCode(chess.PieceType t) {
    switch (t) {
      case chess.PieceType.PAWN:
        return 'p';
      case chess.PieceType.ROOK:
        return 'r';
      case chess.PieceType.KNIGHT:
        return 'n';
      case chess.PieceType.BISHOP:
        return 'b';
      case chess.PieceType.QUEEN:
        return 'q';
      case chess.PieceType.KING:
        return 'k';
      default:
        return 'p';
    }
  }

  String _glyphFromType(String type, chess.Color color) {
    final white = color == chess.Color.WHITE;
    String code;
    switch (type) {
      case 'p':
        code = '♙';
        break;
      case 'r':
        code = '♖';
        break;
      case 'n':
        code = '♘';
        break;
      case 'b':
        code = '♗';
        break;
      case 'q':
        code = '♕';
        break;
      case 'k':
        code = '♔';
        break;
      default:
        code = '♙';
    }
    return white
        ? code
        : ({'♙': '♟', '♖': '♜', '♘': '♞', '♗': '♝', '♕': '♛', '♔': '♚'}[code] ??
            code);
  }

  Widget _capturedRow(List<String> glyphs, {required bool alignEnd}) {
    if (glyphs.isEmpty) return const SizedBox(height: 20);
    return SizedBox(
      height: 24,
      child: Row(
        mainAxisAlignment:
            alignEnd ? MainAxisAlignment.end : MainAxisAlignment.start,
        children: [
          for (final g in glyphs)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 2.0),
              child: Text(g, style: const TextStyle(fontSize: 18)),
            ),
        ],
      ),
    );
  }

  String _resultText() {
    final g = ctrl.game;
    if (_flagged && _winnerOnTime != null) {
      return _winnerOnTime == chess.Color.WHITE
          ? 'White wins on time'
          : 'Black wins on time';
    }
    if (g.in_checkmate) {
      return g.turn == chess.Color.WHITE
          ? 'Black wins by checkmate'
          : 'White wins by checkmate';
    }
    if (g.in_stalemate) return 'Draw by stalemate';
    if (g.in_threefold_repetition) return 'Draw by threefold repetition';
    if (g.insufficient_material) return 'Draw by insufficient material';
    if (g.in_draw) return 'Draw';
    return 'Game over';
  }

  Future<void> _showGameOverDialog() async {
    final msg = _resultText();
    if (!mounted) return;
    await showDialog<void>(
      context: context,
      builder: (_) => AlertDialog(
        title: const Text('Game Over'),
        content: Text(msg),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  Widget _clocksRow() {
    String fmt(Duration? d) {
      if (d == null) return '--:--';
      final m = d.inMinutes;
      final s = d.inSeconds % 60;
      return '${m.toString().padLeft(2, '0')}:${s.toString().padLeft(2, '0')}';
    }

    final active = _sideToMoveClock;
    Color dot(chess.Color c) => active == c ? Colors.green : Colors.grey;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(children: [
            Icon(Icons.circle, size: 10, color: dot(chess.Color.WHITE)),
            const SizedBox(width: 4),
            const Text('White'),
            const SizedBox(width: 6),
            Text(fmt(_timeWhite)),
          ]),
          Row(children: [
            Text(fmt(_timeBlack)),
            const SizedBox(width: 6),
            const Text('Black'),
            const SizedBox(width: 4),
            Icon(Icons.circle, size: 10, color: dot(chess.Color.BLACK)),
          ]),
        ],
      ),
    );
  }
}
