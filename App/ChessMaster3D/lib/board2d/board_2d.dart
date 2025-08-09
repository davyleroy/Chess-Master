import 'package:flutter/material.dart';

class Board2D extends StatelessWidget {
  const Board2D({super.key});

  @override
  Widget build(BuildContext context) {
    final size = MediaQuery.of(context).size;
    final boardSize = size.shortestSide;
    return Center(
      child: SizedBox(
        width: boardSize,
        height: boardSize,
        child: GridView.builder(
          physics: const NeverScrollableScrollPhysics(),
          itemCount: 64,
          gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(crossAxisCount: 8),
          itemBuilder: (_, i) {
            final x = i % 8;
            final y = i ~/ 8;
            final dark = (x + y) % 2 == 1;
            return Container(color: dark ? Colors.brown[700] : Colors.brown[200]);
          },
        ),
      ),
    );
  }
}
